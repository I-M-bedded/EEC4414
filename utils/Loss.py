import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_grid_coords
# feture map의 cell 중심 좌표 원본 이미지 기준으로 계산

def compute_iou_score(pred, target):
    pred_l, pred_t, pred_r, pred_b = pred.unbind(-1)
    target_l, target_t, target_r, target_b = target.unbind(-1)
    
    # 교집합 면적
    w_intersect = torch.min(pred_l, target_l) + torch.min(pred_r, target_r)
    h_intersect = torch.min(pred_t, target_t) + torch.min(pred_b, target_b)
    area_intersect = w_intersect.clamp(min=0) * h_intersect.clamp(min=0)
    
    # 합집합 면적
    pred_area = (pred_l + pred_r) * (pred_t + pred_b)
    target_area = (target_l + target_r) * (target_t + target_b)
    area_union = target_area + pred_area - area_intersect
    
    # IoU (0~1) - 교집합/합집함
    iou = (area_intersect + 1e-7) / (area_union + 1e-7)
    return iou.clamp(min=0, max=1.0)

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

# IoU loss = -log(IoU)
def iou_loss(pred, target):
    iou = compute_iou_score(pred, target)
    return -torch.log(iou + 1e-7)

class FCOSLoss(nn.Module):
    def __init__(self, strides=[4, 8, 16], device='cuda'):
        super().__init__()
        self.strides = strides
        self.device = device
        self.limit_range = torch.tensor([[-1, 64], [64, 128], [128, 99999]], dtype=torch.float32, device=device)

    def forward(self, cls_logits, bbox_reg, targets):
        all_coords = []
        all_level_idx = []
        
        for l, stride in enumerate(self.strides):
            H, W = cls_logits[l].shape[2:]
            coords_x, coords_y = get_grid_coords(H, W, stride, self.device)
            coords = torch.stack([coords_x, coords_y], dim=1) 
            all_coords.append(coords)
            all_level_idx.append(torch.full((len(coords),), l, dtype=torch.long, device=self.device))
            
        all_coords = torch.cat(all_coords, dim=0)
        all_level_idx = torch.cat(all_level_idx, dim=0)
        
        level_low = self.limit_range[all_level_idx, 0]
        level_high = self.limit_range[all_level_idx, 1]

        def flatten_preds(preds):
            flattened = []
            for p in preds:
                flattened.append(p.permute(0, 2, 3, 1).reshape(p.shape[0], -1, p.shape[1]))
            return torch.cat(flattened, dim=1)

        pred_cls = flatten_preds(cls_logits)
        pred_reg = flatten_preds(bbox_reg)

        batch_size = pred_cls.shape[0]
        cls_loss_sum = 0
        reg_loss_sum = 0
        total_num_pos = 0

        for b in range(batch_size):
            gt_boxes = targets[b]['boxes'].to(self.device).float()
            
            # 박스 없으면 배경 학습
            if len(gt_boxes) == 0:
                cls_loss_sum += sigmoid_focal_loss(pred_cls[b], torch.zeros_like(pred_cls[b]))
                continue

            xs, ys = all_coords[:, 0], all_coords[:, 1]
            l = xs[:, None] - gt_boxes[:, 0][None, :]
            t = ys[:, None] - gt_boxes[:, 1][None, :]
            r = gt_boxes[:, 2][None, :] - xs[:, None]
            b_off = gt_boxes[:, 3][None, :] - ys[:, None]
            
            reg_targets_per_im = torch.stack([l, t, r, b_off], dim=2) 
            is_in_box = reg_targets_per_im.min(dim=2)[0] > 0
            max_reg = reg_targets_per_im.max(dim=2)[0]
            is_in_level = (max_reg > level_low[:, None]) & (max_reg < level_high[:, None])

            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            locations_to_gt_area = gt_areas[None, :].repeat(len(all_coords), 1)
            locations_to_gt_area[~(is_in_box & is_in_level)] = float('inf')
            
            min_area, min_gt_idx = locations_to_gt_area.min(dim=1)
            pos_mask = min_area < float('inf')
            
            # 타겟 생성: 물체면 1.0, 아니면 0.0
            cls_target = torch.zeros_like(pred_cls[b])
            cls_target[pos_mask, 0] = 1.0 
            
            num_pos = pos_mask.sum().item()
            total_num_pos += num_pos

            if num_pos > 0:
                curr_reg_pred = pred_reg[b][pos_mask]
                matched_reg_targets = reg_targets_per_im[pos_mask, min_gt_idx[pos_mask]]
                
                # IoU Loss
                iou_l = iou_loss(curr_reg_pred, matched_reg_targets)
                reg_loss_sum += iou_l.sum()

            # Focal Loss
            cls_loss_sum += sigmoid_focal_loss(pred_cls[b], cls_target)
                
        num_pos_avg = max(1.0, total_num_pos / batch_size)
        return (cls_loss_sum + reg_loss_sum) / num_pos_avg, (cls_loss_sum / num_pos_avg, reg_loss_sum / num_pos_avg)