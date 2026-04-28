import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.model import ResNet18

# ============================================
# 1. LOAD DỮ LIỆU
# ============================================
data_dir = "/content/datasets"  # Đường dẫn đến dataset

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Load dataset (nếu có test folder riêng)
test_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "test"), transform=transform
)

# Nếu không có test folder, dùng random_split từ full dataset
if len(test_dataset) == 0:
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    _, _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    # Gán transform cho test set
    test_dataset.dataset.transform = transform

batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"📊 Test samples: {len(test_dataset)}")

# ============================================
# 2. LOAD MODEL
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18(num_classes=2).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

print("✅ Model loaded successfully!")

# ============================================
# 3. TEST
# ============================================
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\n🎯 Test Accuracy: {100 * correct / total:.2f}%")
