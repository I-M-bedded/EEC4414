import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision import tv_tensors  # For wrapping bounding boxes
import urllib.request
import random # For Mosaic augmentation

class BananaDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=(256, 256)):
       
        self.img_dir = os.path.join(root_dir, split, 'images') 
        self.csv_file = os.path.join(root_dir, split, 'label.csv')
        
        if not os.path.exists(self.csv_file):
            print(f"Dataset not found at {root_dir}. Downloading...")
            self._download_dataset(root_dir)
        self.img_size = img_size
        self.mosaic_prob = 0.5  if split == 'bananas_train' else 0.0
        
        # CSV 로드
        # 구조: [filename, label, xmin, ymin, xmax, ymax]
        self.annotations = pd.read_csv(self.csv_file)
        
        # Augmentation (v2 API 사용)
        if split == 'bananas_train':
            self.transforms = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(self.img_size),
                #데이터셋이 바나나가 무작위로 회전해 있는 형상. 과감하게 회전 및 뒤집기 적용
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                
                # Affine 변환: 회전(-30~30도), 이동(10%), 스케일(0.8~1.2배)
                v2.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                
                #색상 변경
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                v2.Normalize(mean=[0.452, 0.456, 0.369], std=[0.224, 0.207, 0.222]), # 바나나 데이터셋 통계 기반 정규화
            ])
        else:
            # Validation은 Resize + Normalize만 적용
            self.transforms = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(self.img_size),
                v2.Normalize(mean=[0.452, 0.456, 0.369], std=[0.224, 0.207, 0.222]), # 바나나 데이터셋 통계 기반 정규화
            ])

    def _download_dataset(self, root_dir):
        dataset_url = 'http://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip'+'5de26c8fce5ccdea9f91267273464dc968d20d72'
        zip_path = os.path.join(root_dir, "banana-detection.zip")
        
        os.makedirs(root_dir, exist_ok=True)
        print(f"Downloading dataset from {dataset_url} to {zip_path}...")
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("Download complete. Please extract the zip file manually.")
        
    def _get_raw_item(self, idx):
        row = self.annotations.iloc[idx]
        img_name = row.iloc[0]
        label = int(row.iloc[1])
        xmin, ymin, xmax, ymax = row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5]

        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path)  # (C, H, W) Tensor
        
        # Bounding Box 포맷팅
        boxes = tv_tensors.BoundingBoxes(
            [[xmin, ymin, xmax, ymax]], 
            format="XYXY", 
            canvas_size=(img.shape[1], img.shape[2])
        )
        labels = torch.tensor([label], dtype=torch.int64)
        
        return img, boxes, labels

    
    # 모자이크 증강
    def mosaic_augmentation(self, index):
        indices = [index] + random.choices(range(len(self)), k=3)
        
        target_w, target_h = self.img_size 
        
        # 빈 캔버스 생성 (C, H, W)
        mosaic_img = torch.zeros((3, target_h, target_w), dtype=torch.float32)
        mosaic_boxes = []
        mosaic_labels = []
        
        # 4분할을 위해 각 이미지를 절반 크기로 리사이즈하는 변환기
        half_w, half_h = target_w // 2, target_h // 2
        resizer = v2.Resize((half_h, half_w)) 

        offsets = [
            (0, 0),          (half_w, 0), 
            (0, half_h),     (half_w, half_h)
        ]
        
        for i, idx in enumerate(indices):
            img, boxes, labels = self._get_raw_item(idx)
            
            # 1) 이미지와 박스를 절반 크기로 리사이즈
            img_resized, boxes_resized = resizer(img, boxes)
            
            # 2) 캔버스에 붙여넣기
            ox, oy = offsets[i]
            mosaic_img[:, oy:oy+half_h, ox:ox+half_w] = img_resized
            
            # 3) 박스 좌표 이동 (Offset 더하기)
            if boxes_resized.numel() > 0:
                # 박스 좌표가 캔버스 기준이 되도록 오프셋 더함
                boxes_resized[:, 0] += ox
                boxes_resized[:, 1] += oy
                boxes_resized[:, 2] += ox
                boxes_resized[:, 3] += oy
                
                mosaic_boxes.append(boxes_resized)
                mosaic_labels.append(labels)
        
        # 4) 박스/라벨 합치기
        if len(mosaic_boxes) > 0:
            mosaic_boxes = torch.cat(mosaic_boxes, dim=0)
            mosaic_labels = torch.cat(mosaic_labels, dim=0)
            
            mosaic_boxes = tv_tensors.BoundingBoxes(
                mosaic_boxes, 
                format="XYXY", 
                canvas_size=(target_h, target_w)
            )
        else:
            # 박스가 없는 경우
            mosaic_boxes = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4)), 
                format="XYXY", 
                canvas_size=(target_h, target_w)
            )
            mosaic_labels = torch.tensor([], dtype=torch.int64)

        return mosaic_img, mosaic_boxes, mosaic_labels
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if random.random() < self.mosaic_prob:
            img, boxes, labels = self.mosaic_augmentation(idx)
        else:
            img, boxes, labels = self._get_raw_item(idx)
        img, boxes = self.transforms(img, boxes)

        return img, {"boxes": boxes, "labels": labels}
    
def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # 이미지는 쌓아서 (B, C, H, W) 형태로 만듦
    images = torch.stack(images, dim=0)
    return images, targets

# 외부에서 호출할 함수
def get_aug_dataloader(root_dir, batch_size=16):
    train_dataset = BananaDataset(root_dir, split='bananas_train')
    val_dataset = BananaDataset(root_dir, split='bananas_val')

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return train_loader, val_loader