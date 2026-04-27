"""
Load datasets progress
Use Yolo to label adult/child by looping through bounding boxes
Chỉ lưu vào datasets/Adult và datasets/Child (không chia train/val/test)
"""

from ultralytics import YOLO
import cv2
import os
import random
import shutil
import json

# ============================================
# CẤU HÌNH
# ============================================
ROOT_INPUT = "root_datasets"
ROOT_OUTPUT = "data"
MODEL_PATH = "yolo26s.pt"
COUNTER_FILE = "counter.json"

# Mapping class ID
CLASS_NAMES = {0: "Adult", 1: "Child"}


# ============================================
# HÀM ĐỌC/GHI COUNTER
# ============================================
def load_counter():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r") as f:
            data = json.load(f)
            return data.get("counter", 1)
    return 1


def save_counter(counter):
    with open(COUNTER_FILE, "w") as f:
        json.dump({"counter": counter}, f)


# ============================================
# HÀM HIỂN THỊ ẢNH
# ============================================
def display_image(window_name, image, size):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    cv2.imshow(window_name, resized)


# ============================================
# TẠO THƯ MỤC OUTPUT
# ============================================
for class_name in CLASS_NAMES.values():
    os.makedirs(os.path.join(ROOT_OUTPUT, class_name), exist_ok=True)

# ============================================
# KIỂM TRA ĐẦU VÀO
# ============================================
if not os.path.exists(ROOT_INPUT):
    print(f"Lỗi: Không tìm thấy thư mục {ROOT_INPUT}!")
    exit()

if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy model {MODEL_PATH}!")
    exit()

# LOAD MODEL YOLO
print("Đang tải model YOLO...")
model = YOLO(MODEL_PATH)

# ============================================
# LẤY DANH SÁCH ẢNH
# ============================================
images = [f for f in os.listdir(ROOT_INPUT) if f.lower().endswith((".jpg"))]
random.shuffle(images)

total = len(images)
print(f"\nTổng số ảnh: {total}")

# ============================================
# THỐNG KÊ VÀ ĐỌC COUNTER
# ============================================
stats = {0: 0, 1: 0}  # {Adult: 0, Child: 0}

start_counter = load_counter()
print(f"\n📌 Số thứ tự ảnh bắt đầu từ: {start_counter}")

# ============================================
# XỬ LÝ TỪNG ẢNH
# ============================================
print(f"\nĐANG XỬ LÝ ẢNH...")
counter = start_counter

for img_name in images:
    img_path = os.path.join(ROOT_INPUT, img_name)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    results = model(img, device=0, classes=[0], conf=0.4)

    has_any_person = False

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
            if crop.size == 0:
                continue

            has_any_person = True

            temp = img.copy()
            cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display_image("full-image", temp, 600)
            display_image("crop-image", crop, 400)

            print(f"\n--- Ảnh: {img_name} ")
            print("Nhấn: [0]=Adult  [1]=Child  [Any key]=Skip  [Q]=Thoát")

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                save_counter(counter)
                print("\n Dừng chương trình")
                exit()
            elif key == ord("0"):
                selected_class = 0
            elif key == ord("1"):
                selected_class = 1
            else:
                print(" Skip.")
                continue

            # save crop image
            ext = img_name.rsplit(".", 1)[-1]
            new_name = f"{counter:05d}.{ext}"

            class_folder = CLASS_NAMES[selected_class]
            dest_dir = os.path.join(ROOT_OUTPUT, class_folder)
            dest_path = os.path.join(dest_dir, new_name)

            cv2.imwrite(dest_path, crop)
            stats[selected_class] += 1

            print(f"Đã lưu: {class_folder}/{new_name}")
            counter += 1

    cv2.destroyAllWindows()

    if has_any_person:
        os.remove(img_path)

    # Lưu counter sau mỗi ảnh
    save_counter(counter)

# ============================================
# LƯU COUNTER LẦN CUỐI
# ============================================
save_counter(counter)
print(f"\n✅ Đã lưu số thứ tự cuối cùng: {counter}")

# ============================================
# THỐNG KÊ DỮ LIỆU
# ============================================
print("\n" + "=" * 50)
print("📊 THỐNG KÊ DỮ LIỆU")
print("=" * 50)
for class_id, class_name in CLASS_NAMES.items():
    print(f"   {class_name}: {stats[class_id]} ảnh")
print(f"   Tổng: {sum(stats.values())} ảnh")

print(f"\n✅ HOÀN THÀNH! Dữ liệu đã sẵn sàng tại: {ROOT_OUTPUT}")
print(f"   {ROOT_OUTPUT}/Adult/")
print(f"   {ROOT_OUTPUT}/Child/")

cv2.destroyAllWindows()
