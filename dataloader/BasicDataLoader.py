import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision import tv_tensors
import urllib.request

class BasicBananaDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=(256, 256)):
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.csv_file = os.path.join(root_dir, split, 'label.csv')
        
        if not os.path.exists(self.csv_file):
            print(f"Dataset not found at {root_dir}. Downloading...")
            self._download_dataset(root_dir)
            
        self.img_size = img_size        
        self.annotations = pd.read_csv(self.csv_file)
        
        # Train과 Val 모두 동일하게 Resize + Normalize만 수행
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True), # 0~255 -> 0.0~1.0
            v2.Resize(self.img_size), #안해도 되지만 일관성 위해 포함
            v2.Normalize(mean=[0.452, 0.456, 0.369], std=[0.224, 0.207, 0.222]), # 바나나 데이터셋 통계 기반 정규화
    ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Pandas 2.0+ 호환성 적용 (iloc 사용)
        row = self.annotations.iloc[idx]
        
        img_name = row.iloc[0]
        label = int(row.iloc[1])
        xmin, ymin, xmax, ymax = row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5]

        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path)

        # Bounding Box Wrapping
        boxes = tv_tensors.BoundingBoxes(
            [[xmin, ymin, xmax, ymax]], 
            format="XYXY", 
            canvas_size=(img.shape[1], img.shape[2])
        )
        labels = torch.tensor([label], dtype=torch.int64)

        # Transform 적용
        img, boxes = self.transforms(img, boxes)

        return img, {"boxes": boxes, "labels": labels}

    def _download_dataset(self, root_dir):
            dataset_url = 'http://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip'+'5de26c8fce5ccdea9f91267273464dc968d20d72'
            zip_path = os.path.join(root_dir, "banana-detection.zip")
            
            os.makedirs(root_dir, exist_ok=True)
            print(f"Downloading dataset from {dataset_url} to {zip_path}...")
            urllib.request.urlretrieve(dataset_url, zip_path)
            print("Download complete. Please extract the zip file manually.")

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets

def get_basic_dataloader(root_dir, batch_size=16):
    # Train셋과 Val셋 모두 BasicBananaDataset 사용
    train_dataset = BasicBananaDataset(root_dir, split='bananas_train')
    val_dataset = BasicBananaDataset(root_dir, split='bananas_val')

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # 순서는 섞지만 데이터 변형은 없음
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader