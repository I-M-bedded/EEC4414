import torch
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from dataloader.AugDataLoader import get_aug_dataloader
from dataloader.BasicDataLoader import get_basic_dataloader
from Final_model import InhaDetector
from utils.Loss import FCOSLoss
from utils.VFL_GIoU_Loss import FCOSLoss_V2

# --- Configuration ---
BATCH_SIZE = 16
LEARNING_RATE = 3e-4 
EPOCHS = 100
SWITCH_EPOCH = 30
FT_EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = './checkpoints'
# --------------------------

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # PHASE 1: Basic DataLoader
    print(f"1: Starting with Basic Loader (Epoch 1 ~ {SWITCH_EPOCH})")
    current_loader, val_loader = get_basic_dataloader(root_dir='data/banana-detection', batch_size=BATCH_SIZE)
    using_aug = False

    model = InhaDetector(num_classes=1).to(DEVICE)
    loss_calc = FCOSLoss_V2(strides=[4, 8, 16], device=DEVICE)
    
    #optimizer 설정 ADAMW LR 3e-4
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 전체 Loss 기록용
    train_losses = []
    val_losses = []
    
    # 세부 Loss 기록용
    train_cls_losses = []
    train_reg_losses = []

    # Best Model Tracking
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        # Phase 2
        if epoch == SWITCH_EPOCH and not using_aug:
            print("\n>> Switching to AugDataLoader (Strong Augmentation) <<")
            current_loader, _ = get_aug_dataloader(root_dir='data/banana-detection', batch_size=BATCH_SIZE)
            using_aug = True
            
        model.train()
        
        # 에폭별 누적 변수 초기화
        running_loss = 0
        running_cls = 0
        running_reg = 0
        
        loop = tqdm(current_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, targets in loop:
            images = images.to(DEVICE)
            
            optimizer.zero_grad()
            
            cls_logits, bbox_reg = model(images)
            
            loss, (l_cls, l_reg) = loss_calc(cls_logits, bbox_reg, targets)
            
            if torch.isnan(loss):
                print("Error: Loss is NaN")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 누적
            running_loss += loss.item()
            running_cls += l_cls.item()
            running_reg += l_reg.item()

            loop.set_postfix(cls=l_cls.item(), reg=l_reg.item())

        scheduler.step()
        
        # 평균 계산 및 저장
        avg_loss = running_loss / len(current_loader)
        avg_cls = running_cls / len(current_loader)
        avg_reg = running_reg / len(current_loader)
        
        train_losses.append(avg_loss)
        train_cls_losses.append(avg_cls)
        train_reg_losses.append(avg_reg)
        
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} (Cls: {avg_cls:.4f}, Reg: {avg_reg:.4f})")

        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'fcos_epoch_{epoch+1}.pth'))
        
        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                
                cls_logits, bbox_reg = model(images)
                loss, _ = loss_calc(cls_logits, bbox_reg, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Best Model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'fcos_best.pth'))
            best_epoch = epoch + 1
            print(f">> {best_epoch} - Best model saved!")
            
    print("\n" + "="*50)
       
    print(f"Phase 3: Fine-tuning with SGD & Basic Data (Epoch {EPOCHS+1} ~ {EPOCHS + FT_EPOCHS})")
    print("Loading Best Model to refine...")
    
    # PHASE 3
    
    # 1. 마지막 에폭 모델 로드
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'fcos_epoch_{EPOCHS}.pth')))
    
    # 2. Optimizer 설정 (SGD, Low LR)
    optimizer_ft = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    
    # 3. 데이터 로더
    ft_loader, _ = get_basic_dataloader(root_dir='data/banana-detection', batch_size=BATCH_SIZE)
    
    
    for epoch in range(EPOCHS, EPOCHS + FT_EPOCHS):
        model.train()
        
        running_loss = 0
        running_cls = 0
        running_reg = 0
        
        #학습상황 플로터
        loop = tqdm(ft_loader, desc=f"Fine-tuning {epoch+1}/{EPOCHS + FT_EPOCHS}")
        
        for images, targets in loop:
            images = images.to(DEVICE)
            
            optimizer_ft.zero_grad()
        
            cls_logits, bbox_reg = model(images)
            
            loss, (l_cls, l_reg) = loss_calc(cls_logits, bbox_reg, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_ft.step()

            running_loss += loss.item()
            running_cls += l_cls.item()
            running_reg += l_reg.item()

            loop.set_postfix(loss=loss.item())
        
        # 평균 계산
        avg_loss = running_loss / len(ft_loader)
        avg_cls = running_cls / len(ft_loader)
        avg_reg = running_reg / len(ft_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                
                cls_logits, bbox_reg = model(images)
                loss, _ = loss_calc(cls_logits, bbox_reg, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"FT Epoch {epoch+1} | Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f}")

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        train_cls_losses.append(avg_cls)
        train_reg_losses.append(avg_reg)

        # Best Model 갱신
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'fcos_best_refined.pth'))
            print(f">> [Refined] Best model updated! ({best_val_loss:.4f})")

    # Plotting
    
    total_epochs_run = len(train_losses)
    epochs_range = range(1, total_epochs_run + 1)
    
    # --- Plot 1: Total Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs_range, val_losses, label='Validation Loss', color='orange')
    
    plt.axvline(x=SWITCH_EPOCH, color='green', linestyle='--', label='Aug Start', alpha=0.5)
    plt.axvline(x=EPOCHS, color='purple', linestyle='--', label='Fine-tuning Start', alpha=0.5)
    
    plt.axvline(x=best_epoch, color='red', linestyle=':', linewidth=2, label=f'Best Epoch ({best_epoch})')
    plt.scatter(best_epoch, best_val_loss, color='red', zorder=5)

    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fcos_total_loss.png')
    print("Saved fcos_total_loss.png")
    
    # --- Plot 2: Component Losses ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs_range, train_cls_losses, label='Cls Loss', color='red')
    plt.plot(epochs_range, train_reg_losses, label='Reg Loss', color='blue')
    
    plt.axvline(x=SWITCH_EPOCH, color='black', linestyle='--', label='Aug Start', alpha=0.3)
    plt.axvline(x=EPOCHS, color='purple', linestyle='--', label='Fine-tuning Start', alpha=0.3)

    plt.xlabel('Epochs')
    plt.ylabel('Component Loss')
    plt.title('Training Component Losses Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log') 
    
    plt.savefig('fcos_component_loss.png')
    print("Saved fcos_component_loss.png")

if __name__ == "__main__":
    train()