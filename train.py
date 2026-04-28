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
# HÀM LOAD DỮ LIỆU
# ============================================
def get_data_loaders(data_dir, batch_size=64, num_workers=2):
    """Tạo DataLoader cho train, val, test"""

    transform_train = transforms.Compose(
        [
            # 1. Resize lớn hơn trước khi crop (tăng đa dạng)
            transforms.Resize((256, 256)),
            # 2. Random crop về 224 (mỗi lần cắt 1 vùng khác nhau)
            transforms.RandomCrop(224),
            # 3. Lật ngang (50%)
            transforms.RandomHorizontalFlip(p=0.5),
            # 4. Xoay (tăng góc lên 20 độ)
            transforms.RandomRotation(degrees=20),
            # 5. Thay đổi màu sắc (tăng cường)
            transforms.ColorJitter(
                brightness=0.3,  # Độ sáng (±30%)
                contrast=0.3,  # Độ tương phản (±30%)
                saturation=0.2,  # Độ bão hòa (±20%)
                hue=0.1,  # Màu sắc (±10%)
            ),
            # 6. Chuyển sang ảnh xám ngẫu nhiên (10%)
            transforms.RandomGrayscale(p=0.1),
            # 7. Chuyển thành tensor
            transforms.ToTensor(),
            # 8. Chuẩn hóa
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # 9. Xóa ngẫu nhiên 1 vùng hình chữ nhật (20%) - giống Cutout
            transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
            ),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"), transform=transform_train
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"), transform=transform_val
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"), transform=transform_val
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


# ============================================
# HÀM TRAIN MỘT EPOCH
# ============================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc="[Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(
            {"loss": loss.item(), "acc": 100.0 * train_correct / train_total}
        )

    return train_loss / len(train_loader), 100.0 * train_correct / train_total


# ============================================
# HÀM VALIDATION
# ============================================
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Val]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    return val_loss / len(val_loader), 100.0 * val_correct / val_total


# ============================================
# CODE CHÍNH (CHỈ CHẠY KHI GỌI TRỰC TIẾP)
# ============================================
if __name__ == "__main__":
    # ===== 1. CẤU HÌNH =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = "datasets"
    batch_size = 64
    num_epochs = 60
    learning_rate = 0.0001
    weight_decay = 0.0001

    # ===== 2. LOAD DỮ LIỆU =====
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        data_dir, batch_size, num_workers=2
    )

    print(f"\n📊 DỮ LIỆU:")
    print(f"   Train: {len(train_loader.dataset)} ảnh")
    print(f"   Val: {len(val_loader.dataset)} ảnh")
    print(f"   Test: {len(test_loader.dataset)} ảnh")
    print(f"   Classes: {classes}")

    # ===== 3. KHỞI TẠO MODEL =====
    model = ResNet18(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # ===== 4. TRAINING LOOP =====
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("\n" + "=" * 50)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN")
    print("=" * 50)

    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # In kết quả
        print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   Learning Rate: {learning_rate}")

        # Lưu model tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet_model.pth")
            print(f"   ✅ Đã lưu model mới (Val Acc: {val_acc:.2f}%)")

        print("-" * 50)

    # ===== 5. TEST =====
    print("\n" + "=" * 50)
    print("📊 ĐÁNH GIÁ TRÊN TẬP TEST")
    print("=" * 50)

    model.eval()
    test_correct, test_total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="[Test]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * test_correct / test_total
    print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")

    # ===== 6. VẼ BIỂU ĐỒ =====
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

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
