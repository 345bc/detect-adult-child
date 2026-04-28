"""
Bước 3: Huấn luyện ResNet-18 (Loss Function + Optimizer + Train Loop)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import model từ Bước 2
from models.model import ResNet18

# ============================================
# 1. KHỞI TẠO MODEL, LOSS, OPTIMIZER
# ============================================

# Tạo model
model = ResNet18(num_classes=2)

# Đưa model lên GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Loss function (chỉ 1 lần)
criterion = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 0.0001
weight_decay = 0.0001

optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)


# ============================================
# 2. LOAD DỮ LIỆU
# ============================================
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Datasets path
data_dir = "/content/drive/MyDrive/datasets"

# Transform cho ảnh
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load dataset
train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "train"), transform=transform_train
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "val"), transform=transform
)
test_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "test"), transform=transform
)

# Tạo DataLoader
batch_size = 64
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

print(f"\n📊 DỮ LIỆU:")
print(f"   Train: {len(train_dataset)} ảnh")
print(f"   Val: {len(val_dataset)} ảnh")
print(f"   Test: {len(test_dataset)} ảnh")
print(f"   Classes: {train_dataset.classes}")

# ============================================
# 3. VÒNG LẶP HUẤN LUYỆN
# ============================================

num_epochs = 100  # Số epoch
best_val_acc = 0

# Lưu lịch sử để vẽ biểu đồ
train_losses = []
val_losses = []
train_accs = []
val_accs = []

print("\n" + "=" * 50)
print("🚀 BẮT ĐẦU HUẤN LUYỆN")
print("=" * 50)

for epoch in range(num_epochs):
    # ===== TRAIN =====
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Thống kê
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        # Cập nhật thanh tiến trình
        pbar.set_postfix(
            {"loss": loss.item(), "acc": 100.0 * train_correct / train_total}
        )

    train_acc = 100.0 * train_correct / train_total
    train_loss_avg = train_loss / len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100.0 * val_correct / val_total
    val_loss_avg = val_loss / len(val_loader)

    # Lưu lịch sử
    train_losses.append(train_loss_avg)
    val_losses.append(val_loss_avg)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Cập nhật scheduler
    # scheduler.step()

    # In kết quả
    print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
    print(f"   Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"   Learning Rate: {learning_rate}")

    # Lưu model tốt nhất
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_resnet_model.pth")
        print(f"   ✅ Đã lưu model mới (Val Acc: {val_acc:.2f}%)")

    print("-" * 50)

# ============================================
# 4. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST
# ============================================

print("\n" + "=" * 50)
print("📊 ĐÁNH GIÁ TRÊN TẬP TEST")
print("=" * 50)

model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = 100.0 * test_correct / test_total
print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")
print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")

# ============================================
# 5. VẼ BIỂU ĐỒ
# ============================================

plt.figure(figsize=(12, 4))

# Biểu đồ Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curves")

# Biểu đồ Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy Curves")

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

print("\n" + "=" * 50)
print("🎉 HOÀN THÀNH HUẤN LUYỆN!")
print("=" * 50)
print(f"📁 Model đã lưu: best_resnet_model.pth")
print(f"📁 Biểu đồ: training_curves.png")
