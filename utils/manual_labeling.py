"""
QUY TRÌNH XỬ LÝ DỮ LIỆU CHO RESNET
Dùng YOLO phát hiện + gán nhãn Adult/Child → Xuất thẳng ra format ResNet
"""

from ultralytics import YOLO
import cv2
import os
import random
import shutil

# ============================================
# CẤU HÌNH (BẠN SỬA LẠI ĐƯỜNG DẪN CHO ĐÚNG)
# ============================================
ROOT_INPUT = "dataroot"              # Thư mục chứa ảnh gốc
ROOT_OUTPUT = "data_processed"       # Thư mục output cho ResNet
MODEL_PATH = "yolo26s.pt"            # File model YOLO

# Tỷ lệ chia dữ liệu
TRAIN_RATIO = 0.7    # 70% train
VAL_RATIO = 0.2      # 20% validation  
TEST_RATIO = 0.1     # 10% test

# Mapping class ID -> tên thư mục
CLASS_NAMES = {0: "Adult", 1: "Child"}

# ============================================
# HÀM HIỂN THỊ ẢNH
# ============================================
def show_keep_ratio(window_name, image, target_size=600):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    cv2.imshow(window_name, resized)

# ============================================
# TẠO THƯ MỤC OUTPUT
# ============================================
for split in ['train', 'val', 'test']:
    for class_name in CLASS_NAMES.values():
        os.makedirs(os.path.join(ROOT_OUTPUT, split, class_name), exist_ok=True)

# ============================================
# KIỂM TRA ĐẦU VÀO
# ============================================
if not os.path.exists(ROOT_INPUT):
    print(f"❌ Lỗi: Không tìm thấy thư mục {ROOT_INPUT}!")
    exit()

if not os.path.exists(MODEL_PATH):
    print(f"❌ Lỗi: Không tìm thấy model {MODEL_PATH}!")
    exit()

# ============================================
# LOAD MODEL YOLO
# ============================================
print("🚀 Đang tải model YOLO...")
model = YOLO(MODEL_PATH)

# ============================================
# LẤY DANH SÁCH ẢNH VÀ CHIA SPLIT
# ============================================
images = [f for f in os.listdir(ROOT_INPUT) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(images)

total = len(images)
train_end = int(TRAIN_RATIO * total)
val_end = int((TRAIN_RATIO + VAL_RATIO) * total)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:]
}

print(f"\n📊 Tổng số ảnh: {total}")
print(f"   Train: {len(splits['train'])} ảnh")
print(f"   Val: {len(splits['val'])} ảnh")
print(f"   Test: {len(splits['test'])} ảnh")

# ============================================
# THỐNG KÊ
# ============================================
stats = {split: {0: 0, 1: 0} for split in splits.keys()}

# ============================================
# XỬ LÝ TỪNG ẢNH
# ============================================
for split, img_list in splits.items():
    print(f"\n📂 ĐANG XỬ LÝ TẬP: {split.upper()}")
    counter = 1
    
    for img_name in img_list:
        img_path = os.path.join(ROOT_INPUT, img_name)
        
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w, _ = img.shape
        results = model(img, device=0, classes=[0], conf=0.4)
        
        has_label = False
        selected_class = None
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                if crop.size == 0:
                    continue
                
                temp = img.copy()
                cv2.rectangle(temp, (x1,y1), (x2,y2), (0,255,0), 2)
                
                show_keep_ratio("FULL IMAGE - GREEN BOX", temp, 800)
                show_keep_ratio("CROP - 0:Adult | 1:Child", crop, 400)
                
                print(f"\n--- [{split}] Ảnh {counter}/{len(img_list)}: {img_name} ---")
                print("Nhấn: [0]=Adult  [1]=Child  [S]=Skip  [Q]=Thoát")
                
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    print("\n❌ Dừng theo yêu cầu.")
                    exit()
                elif key == ord('0'):
                    selected_class = 0
                    has_label = True
                elif key == ord('1'):
                    selected_class = 1
                    has_label = True
                else:
                    print("⏭️ Bỏ qua box này.")
                    continue
                
                break  # Chỉ lấy box đầu tiên
        
        cv2.destroyAllWindows()
        
        # Lưu ảnh vào đúng thư mục theo class
        if has_label and selected_class is not None:
            ext = img_name.rsplit('.', 1)[-1]
            new_name = f"{split}_{counter:05d}.{ext}"
            
            class_folder = CLASS_NAMES[selected_class]
            dest_dir = os.path.join(ROOT_OUTPUT, split, class_folder)
            dest_path = os.path.join(dest_dir, new_name)
            
            shutil.copy2(img_path, dest_path)
            stats[split][selected_class] += 1
            
            print(f"✅ Đã lưu: {split}/{class_folder}/{new_name}")
            counter += 1

# ============================================
# IN KẾT QUẢ THỐNG KÊ
# ============================================
print("\n" + "="*60)
print("📊 THỐNG KÊ DỮ LIỆU CHO RESNET")
print("="*60)

for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()}:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"   - {class_name}: {stats[split][class_id]} ảnh")
    print(f"   Tổng: {sum(stats[split].values())} ảnh")

print("\n" + "="*60)
print(f"🎉 HOÀN THÀNH! Dữ liệu đã sẵn sàng tại: {ROOT_OUTPUT}")
print("="*60)

cv2.destroyAllWindows()