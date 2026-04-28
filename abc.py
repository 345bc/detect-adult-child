"""
test_augmentation.py - Kiểm tra ảnh sau khi augmentation
Xem ảnh gốc và ảnh đã qua augmentation để đánh giá
"""

import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# ============================================
# CẤU HÌNH AUGMENTATION
# ============================================
transform_train = transforms.Compose(
    [
        transforms.Resize((280, 280)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
        transforms.RandomGrayscale(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ]
)

# Transform KHÔNG augmentation (để so sánh)
transform_none = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# ============================================
# HÀM HIỂN THỊ ẢNH
# ============================================
def tensor_to_image(tensor):
    """Chuyển tensor về numpy array để hiển thị"""
    img = tensor.numpy().transpose(1, 2, 0)
    # Denormalize (chỉ để hiển thị, không dùng cho train)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


# ============================================
# TEST ẢNH
# ============================================
def test_augmentation(image_path, num_samples=5):
    """Hiển thị ảnh gốc và các biến thể augmentation"""

    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    # Đọc ảnh gốc
    original = Image.open(image_path).convert("RGB")

    # Tạo các biến thể
    samples = [original]
    for _ in range(num_samples):
        samples.append(transform_train(original))

    # Hiển thị
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 5))

    # Ảnh gốc
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Các biến thể
    for i in range(num_samples):
        img = tensor_to_image(samples[i + 1])
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Augmented {i+1}")
        axes[i + 1].axis("off")

    plt.suptitle(f"Augmentation Demo - {os.path.basename(image_path)}")
    plt.tight_layout()
    plt.show()

    print(f"✅ Hiển thị ảnh gốc và {num_samples} biến thể augmentation")


# ============================================
# TEST NHIỀU ẢNH
# ============================================
def test_batch(folder_path, num_images=3, num_samples=3):
    """Test augmentation trên nhiều ảnh trong folder"""

    # Lấy danh sách ảnh
    extensions = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

    if not images:
        print(f"❌ No images found in {folder_path}")
        return

    print(f"📁 Found {len(images)} images")

    # Test trên vài ảnh
    for img_name in images[:num_images]:
        img_path = os.path.join(folder_path, img_name)
        test_augmentation(img_path, num_samples)


# ============================================
# TEST SO SÁNH NHIỀU AUGMENTATION MỨC ĐỘ
# ============================================
def compare_augmentation_levels(image_path):
    """So sánh các mức độ augmentation khác nhau"""

    levels = {
        "Level 1 (Nhẹ)": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        ),
        "Level 2 (Trung bình)": transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ]
        ),
        "Level 3 (Mạnh - hiện tại)": transforms.Compose(
            [
                transforms.Resize((280, 280)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=25),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15
                ),
                transforms.RandomGrayscale(p=0.15),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.25),
            ]
        ),
    }

    original = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Dòng 1: Ảnh gốc + các biến thể
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    for i, (name, transform) in enumerate(levels.items()):
        aug_img = transform(original)
        if isinstance(aug_img, torch.Tensor):
            aug_img = tensor_to_image(aug_img)
        axes[0, i + 1].imshow(aug_img)
        axes[0, i + 1].set_title(name)
        axes[0, i + 1].axis("off")

    # Dòng 2: Nhiều biến thể của Level 3
    axes[1, 0].axis("off")
    for i in range(2):
        aug_img = levels["Level 3 (Mạnh - hiện tại)"](original)
        aug_img = tensor_to_image(aug_img)
        axes[1, i + 1].imshow(aug_img)
        axes[1, i + 1].set_title(f"Level 3 - Var {i+1}")
        axes[1, i + 1].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # ===== CẤU HÌNH =====
    # Cách 1: Test 1 ảnh cụ thể
    IMAGE_PATH = "test/images/test_image_00001.jpg"  # ← SỬA ĐƯỜNG DẪN

    # Cách 2: Test cả folder
    # FOLDER_PATH = "datasets/train/Adult"  # ← SỬA ĐƯỜNG DẪN

    print("=" * 50)
    print("🖼️ AUGMENTATION VISUALIZATION TOOL")
    print("=" * 50)

    # Test 1 ảnh
    if os.path.exists(IMAGE_PATH):
        print(f"\n📸 Testing single image: {IMAGE_PATH}")
        test_augmentation(IMAGE_PATH, num_samples=6)

        print("\n📊 Comparing augmentation levels...")
        compare_augmentation_levels(IMAGE_PATH)
    else:
        print(f"❌ Image not found: {IMAGE_PATH}")
        print("Please update IMAGE_PATH in the script")

    # Test cả folder (tùy chọn)
    # if os.path.exists(FOLDER_PATH):
    #     print(f"\n📁 Testing folder: {FOLDER_PATH}")
    #     test_batch(FOLDER_PATH, num_images=3, num_samples=3)
