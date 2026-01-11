import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_grid_coords

# IoU 계산기
def compute_iou_score(pred, target):
    pred_l, pred_t, pred_r, pred_b = pred.unbind(-1)
    target_l, target_t, target_r, target_b = target.unbind(-1)
    
    w_intersect = torch.min(pred_l, target_l) + torch.min(pred_r, target_r)
    h_intersect = torch.min(pred_t, target_t) + torch.min(pred_b, target_b)
    area_intersect = w_intersect.clamp(min=0) * h_intersect.clamp(min=0)
    
    pred_area = (pred_l + pred_r) * (pred_t + pred_b)
    target_area = (target_l + target_r) * (target_t + target_b)
    area_union = target_area + pred_area - area_intersect + 1e-7
    
    return (area_intersect / area_union).clamp(min=0, max=1.0)

# GIoU Loss
def giou_loss(pred, target):
    pred_l, pred_t, pred_r, pred_b = pred.unbind(-1)
    target_l, target_t, target_r, target_b = target.unbind(-1)
    
    # 1. IoU 계산
    w_intersect = torch.min(pred_l, target_l) + torch.min(pred_r, target_r)
    h_intersect = torch.min(pred_t, target_t) + torch.min(pred_b, target_b)
    area_intersect = w_intersect.clamp(min=0) * h_intersect.clamp(min=0)
    
    pred_area = (pred_l + pred_r) * (pred_t + pred_b)
    target_area = (target_l + target_r) * (target_t + target_b)
    area_union = pred_area + target_area - area_intersect + 1e-7
    iou = area_intersect / area_union

    # 2. 외접 사각형
    w_enclose = torch.max(pred_l, target_l) + torch.max(pred_r, target_r)
    h_enclose = torch.max(pred_t, target_t) + torch.max(pred_b, target_b)
    area_enclose = w_enclose.clamp(min=0) * h_enclose.clamp(min=0) + 1e-7
    
    # 3. GIoU Term
    giou = iou - (area_enclose - area_union) / area_enclose
    return 1.0 - giou

# Varifocal Loss
def varifocal_loss(pred_logits, gt_iou, alpha=0.75, gamma=2.0):
    pred_score = pred_logits.sigmoid()
    pos_mask = (gt_iou > 0).float()
    
    # Positive: -q * (q * log(p) + (1-q) * log(1-p)) (Asymmetric)
    pos_loss = -gt_iou * (gt_iou * pred_score.log() + (1 - gt_iou) * (1 - pred_score).log())
    
    # Negative: -alpha * p^gamma * log(1-p) (Focal)
    neg_loss = -alpha * (pred_score ** gamma) * (1 - pred_score).log()
    
    loss = pos_mask * pos_loss + (1 - pos_mask) * neg_loss
    return loss.sum()

class FCOSLoss_V2(nn.Module):
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
            
            if len(gt_boxes) == 0:
                # 배경 학습 (VFL은 target=0이면 Negative로 작동)
                cls_loss_sum += varifocal_loss(pred_cls[b], torch.zeros_like(pred_cls[b]))
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
            
            # 초기 타겟 0으로 생성
            cls_target = torch.zeros_like(pred_cls[b])
            
            num_pos = pos_mask.sum().item()
            total_num_pos += num_pos

            if num_pos > 0:
                curr_reg_pred = pred_reg[b][pos_mask]
                matched_reg_targets = reg_targets_per_im[pos_mask, min_gt_idx[pos_mask]]
                
                # Dynamic IoU Score 계산 (Soft Label)
                iou_scores = compute_iou_score(curr_reg_pred, matched_reg_targets).detach()
                cls_target[pos_mask, 0] = iou_scores
                
                # GIoU Loss 사용
                loss_giou = giou_loss(curr_reg_pred, matched_reg_targets)
                reg_loss_sum += loss_giou.sum()

            # Varifocal Loss 사용
            cls_loss_sum += varifocal_loss(pred_cls[b], cls_target)
                
        num_pos_avg = max(1.0, total_num_pos / batch_size)
        return (cls_loss_sum + reg_loss_sum) / num_pos_avg, (cls_loss_sum / num_pos_avg, reg_loss_sum / num_pos_avg)