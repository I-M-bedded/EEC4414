import torch
import cv2
import numpy as np
import torchvision.transforms.v2 as v2
from torchvision.ops import nms
import matplotlib.pyplot as plt
import os
from Final_model import InhaDetector
from utils.utils import get_grid_coords, decode_fcos_boxes

# --- Configuration ---
MODEL_PATH = './checkpoints/fcos_best.pth'  # 최적 모델 경로
MODEL_PATH_30 = './checkpoints/fcos_epoch_30.pth'
MODEL_PATH_50 = './checkpoints/fcos_epoch_50.pth'
MODEL_PATH_70 = './checkpoints/fcos_epoch_70.pth'
MODEL_PATH_100 = './checkpoints/fcos_epoch_100.pth'
MODEL_PATH_150 = './checkpoints/fcos_epoch_150.pth'
IMG_PATH = './banana.jpg'
IMG_SIZE = (256, 256)

# [NMS 설정]
CONF_THRESHOLD = 0.3    # 1차 필터링
IOU_THRESHOLD = 0.25    # NMS 임계값
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference(model_path=MODEL_PATH, save_path='result_fcos.jpg'):
    print(f"Loading model from {model_path}...")
    
    model = InhaDetector(num_classes=1).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    except:
        print("Warning: Loading with weights_only=False")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
    model.eval()

    # 1. Image Load & Preprocess
    if not os.path.exists(IMG_PATH):
        print(f"Error: Image not found at {IMG_PATH}")
        return

    original_img = cv2.imread(IMG_PATH)    
    orig_h, orig_w = original_img.shape[:2]
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(IMG_SIZE),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(original_img_rgb).unsqueeze(0).to(DEVICE)

    # 2. Forward Pass
    strides = [4, 8, 16]
    
    with torch.no_grad():
        # 리턴값 2개 (cls, reg)
        cls_logits, bbox_reg = model(input_tensor)
        
        all_boxes = []
        all_scores = []
        
        for l, stride in enumerate(strides):
            cls_score = cls_logits[l][0]   # (C, H, W)
            reg_map = bbox_reg[l][0]       # (4, H, W)
            # cnt_score 삭제됨
            
            _, H, W = cls_score.shape
            
            # 1) 차원 변경
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, 1).sigmoid()
            reg_map = reg_map.permute(1, 2, 0).reshape(-1, 4)
            
            # 2) Grid 생성
            coords_x, coords_y = get_grid_coords(H, W, stride, DEVICE)
            coords = torch.stack([coords_x, coords_y], dim=1) 
            
            # 3) [수정] 점수 계산: Centerness 곱하기 없이 오직 Cls Score만 사용
            final_scores = cls_score
            
            # 4) 1차 필터링
            keep_mask = final_scores.squeeze() > CONF_THRESHOLD
            
            if keep_mask.sum() == 0:
                continue
                
            valid_scores = final_scores[keep_mask]
            valid_reg = reg_map[keep_mask]
            valid_coords = coords[keep_mask]
            
            # 5) 박스 디코딩
            decoded_boxes = decode_fcos_boxes(valid_reg, valid_coords)
            
            # 6) 클리핑
            decoded_boxes[:, 0].clamp_(min=0, max=IMG_SIZE[1])
            decoded_boxes[:, 1].clamp_(min=0, max=IMG_SIZE[0])
            decoded_boxes[:, 2].clamp_(min=0, max=IMG_SIZE[1])
            decoded_boxes[:, 3].clamp_(min=0, max=IMG_SIZE[0])

            all_boxes.append(decoded_boxes)
            all_scores.append(valid_scores)
    
    # 3. NMS 수행
    if not all_boxes:
        print("No banana detected.")
        return

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0).squeeze()

    print(f"Candidates before NMS: {len(all_boxes)}")

    keep_indices = nms(all_boxes, all_scores, IOU_THRESHOLD)
    
    final_boxes = all_boxes[keep_indices]
    final_scores = all_scores[keep_indices]

    print(f"Final Detections after NMS: {len(final_boxes)}")

    # 4. Visualization
    scale_x = orig_w / IMG_SIZE[0]
    scale_y = orig_h / IMG_SIZE[1]
    draw_img = original_img.copy()

    for box, score in zip(final_boxes, final_scores):
        x1, y1, x2, y2 = box.cpu().numpy()
        
        # 원본 해상도로 복구
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        # 박스 그리기
        cv2.rectangle(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        label_text = f"{score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (t_w, t_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Y 좌표 보정 (위쪽 잘림 방지)
        if y1 - t_h - 5 < 0:
            text_y = int(y1 + t_h + 10) # 박스 안쪽
        else:
            text_y = int(y1 - 5)        # 박스 바깥

        if x1 + t_w > orig_w:
            text_x = int(orig_w - t_w - 5)
        else:
            text_x = int(x1)

        cv2.putText(draw_img, label_text, (text_x, text_y), 
                    font, font_scale, (0, 255, 0), thickness)

    
    cv2.imwrite(save_path, draw_img)
    print(f"Inference result saved as '{save_path}'")

if __name__ == "__main__":
    inference()
    
    # 모델별 비교 실행
    if os.path.exists(MODEL_PATH_30):
        inference(MODEL_PATH_30, 'result_fcos_30.jpg')
    if os.path.exists(MODEL_PATH_50):
        inference(MODEL_PATH_30, 'result_fcos_50.jpg')
    if os.path.exists(MODEL_PATH_70):
        inference(MODEL_PATH_70, 'result_fcos_70.jpg')
    if os.path.exists(MODEL_PATH_100):
        inference(MODEL_PATH_100, 'result_fcos_100.jpg')