"""
load datasets progress
Use Yolo to label adult/child by looping through bounding boxes
"""

from ultralytics import YOLO
import cv2
import os
import random
import shutil

# Configuration
ROOT_INPUT = "dataroot"
ROOT_OUTPUT = "datasets"
MODEL_PATH = "yolo26s.pt"

# Spit data into tran/val/test
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Mapping class ID
CLASS_NAMES = {0: "Adult", 1: "Child"}


# Display images function
def display_image(window_name, image, size):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    cv2.imshow(window_name, resized)


# Output folder
for split in ["train", "val", "test"]:
    for class_name in CLASS_NAMES.values():
        os.makedirs(os.path.join(ROOT_OUTPUT, split, class_name), exist_ok=True)


# Validation input
if not os.path.exists(ROOT_INPUT):
    print(f"Lỗi: Không tìm thấy thư mục {ROOT_INPUT}!")
    exit()

if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy model {MODEL_PATH}!")
    exit()


# Load Yolo model
print("Đang tải model YOLO...")
model = YOLO(MODEL_PATH)


# Get list of images and split
images = [f for f in os.listdir(ROOT_INPUT) if f.lower().endswith((".jpg"))]
random.shuffle(images)

total = len(images)
train_end = int(TRAIN_RATIO * total)
val_end = int((TRAIN_RATIO + VAL_RATIO) * total)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:],
}

print(f"\nTổng số ảnh: {total}")
print(f"Training_set: {len(splits['train'])} ảnh")
print(f"Validation_set: {len(splits['val'])} ảnh")
print(f"Testing_set: {len(splits['test'])} ảnh")

# stats
stats = {}
for split in splits.keys():
    stats[split] = {0: 0, 1: 0}

# image processing
for split, img_list in splits.items():
    print(f"\nĐANG XỬ LÝ TẬP: {split}")
    counter = 1

    for img_name in img_list:
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

                # Display full image
                temp = img.copy()
                cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                display_image("full-image", temp, 600)

                # Display crop image
                display_image("crop-image", crop, 400)

                print(f"\n--- [{split}] Ảnh: {img_name} ")
                print("Nhấn: [0]=Adult  [1]=Child  [Any key]=Skip  [Q]=Thoát")

                # Wait for user to press a key
                key = cv2.waitKey(0) & 0xFF

                if key == ord("q"):
                    cv2.destroyAllWindows()
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
                new_name = f"{split}_{counter:05d}.{ext}"

                class_folder = CLASS_NAMES[selected_class]
                dest_dir = os.path.join(ROOT_OUTPUT, split, class_folder)
                dest_path = os.path.join(dest_dir, new_name)

                cv2.imwrite(dest_path, crop)
                stats[split][selected_class] += 1

                print(f"Đã lưu: {split}/{class_folder}/{new_name}")
                counter += 1

        cv2.destroyAllWindows()

        if has_any_person:
            os.remove(img_path)


# Stat datasets
print("THỐNG KÊ DỮ LIỆU")

for split in splits:
    print(f"\n{split}:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"   - {class_name}: {stats[split][class_id]} ảnh")
    print(f"   Tổng: {sum(stats[split].values())} ảnh")

print(f"HOÀN THÀNH! Dữ liệu đã sẵn sàng tại: {ROOT_OUTPUT}")

cv2.destroyAllWindows()
