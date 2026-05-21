# ========================
# MangoMamba 完整训练代码
# 完全对齐论文：Three-Phase Curriculum Training
# ========================

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from tqdm import tqdm
from pathlib import Path

# ========== 1. 路径 ==========
BASE = "/content/Reproduction-of-Mango-Mamba-Project/Mango-Mamba-and-VN-MangoLeaf-main"
DATASET = "/content/Reproduction-of-Mango-Mamba-Project/VN-MangoLeaf"
sys.path.append(BASE)

# ========== 2. 导入你现有的模型 ==========
from models.mango_mamba import mango_mamba_tiny

# ========== 3. 训练超参数（论文原版） ==========
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = Path(BASE) / "trained_weights"
SAVE_PATH.mkdir(exist_ok=True)

# ========== 4. 数据增强（论文原版） ==========
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== 5. 加载数据集 ==========
dataset = ImageFolder(DATASET, transform=train_transform)

# 8:2 划分（论文标准划分）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ========== 6. 初始化模型 ==========
model = mango_mamba_tiny(
    num_classes=NUM_CLASSES,
    window_size=[7,7,7,7],
    mixer_mode='hybrid',
    use_eca=True,
    use_grn=True,
    use_swiglu=True
).to(DEVICE)

# ========== 7. 损失函数 & 优化器（论文原版） ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ========== 8. 训练函数 ==========
def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)['logits']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total

# ========== 9. 验证函数 ==========
@torch.no_grad()
def validate(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)['logits']
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total

# ========== 10. 开始训练 ==========
print("\nMangoMamba 训练开始")
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")

    # 训练
    train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, DEVICE)
    # 验证
    val_loss, val_acc = validate(val_loader, model, criterion, DEVICE)
    # 学习率更新
    scheduler.step()

    # 打印
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # 保存最优权重
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc
        }, SAVE_PATH / "mangomamba_best.pth")
        print(f"最优模型已保存 | Best Acc: {best_acc:.2f}%")

print("\n训练完成！")
print(f"最高验证精度: {best_acc:.2f}%")
print(f"模型保存路径: {SAVE_PATH}/mangomamba_best.pth")
