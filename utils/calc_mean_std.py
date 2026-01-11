from dataloader.BasicDataLoader import get_basic_dataloader
from tqdm import tqdm

def calculate_dataset_stats():
    train_loader, _ = get_basic_dataloader(root_dir='data/banana-detection', batch_size=64)
    
    mean = 0.
    std = 0.
    total_images_count = 0
    
    print("평균 표준 편차 계산 중")
    for images, _ in tqdm(train_loader):
        # images shape: [Batch, 3, H, W]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1) # [Batch, 3, H*W]
        
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    print(f"\n이미지 평균: {mean.tolist()}")
    print(f"이미지 표준편차:  {std.tolist()}")
    
    return mean, std

if __name__ == "__main__":
    calculate_dataset_stats()