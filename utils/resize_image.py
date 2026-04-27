"""
Resize tất cả ảnh trong folder về 224x224 (giữ tỉ lệ, thêm viền đen)
"""

import cv2
import numpy as np
import os
from tqdm import tqdm


INPUT_DIR = "Datasets/val/Child"
OUTPUT_DIR = "Child"
TARGET_SIZE = (224, 224)


# ============================================
# HÀM RESIZE
# ============================================
def resize_with_padding(image, target_size):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


# ============================================
# THỰC THI
# ============================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(extensions)]

print(f"Tìm thấy {len(images)} ảnh trong {INPUT_DIR}")
print(f"Resize về {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

for img_name in tqdm(images):
    input_path = os.path.join(INPUT_DIR, img_name)
    output_path = os.path.join(OUTPUT_DIR, img_name)

    img = cv2.imread(input_path)
    if img is None:
        print(f"Không đọc được: {img_name}")
        continue

    resized = resize_with_padding(img, TARGET_SIZE)
    cv2.imwrite(output_path, resized)

print(f"✅ Hoàn thành! Ảnh đã lưu tại: {OUTPUT_DIR}")
