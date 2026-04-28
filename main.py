"""
predict.py - Dự đoán Adult/Child bằng ResNet-18
Hỗ trợ: Ảnh, Video, Webcam
"""

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from models.model import ResNet18

# ============================================
# CẤU HÌNH
# ============================================
MODEL_PATH = r"best_resnet_model.pth"
CLASS_NAMES = ["Adult", "Child"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform cho ảnh đầu vào
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ============================================
# LOAD MODEL RESNET
# ============================================
def load_model():
    model = ResNet18(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✅ Loaded model: {MODEL_PATH}")
    print(f"✅ Device: {DEVICE}")
    return model


# ============================================
# DỰ ĐOÁN 1 ẢNH (numpy array)
# ============================================
def predict_frame(model, frame):
    """
    frame: numpy array (H, W, 3) từ cv2 (BGR)
    return: class_name, confidence
    """
    # Chuyển BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Chuyển sang PIL
    pil_img = Image.fromarray(rgb_frame)

    # Transform
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    class_name = CLASS_NAMES[pred.item()]
    confidence = confidence.item()

    return class_name, confidence


# ============================================
# XỬ LÝ ẢNH
# ============================================
def process_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    class_name, confidence = predict_frame(model, frame)

    # Vẽ kết quả
    color = (0, 255, 0) if class_name == "Adult" else (0, 0, 255)
    label = f"{class_name}: {confidence*100:.1f}%"

    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    print(f"📊 {os.path.basename(image_path)} → {label}")

    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================
# XỬ LÝ VIDEO / WEBCAM
# ============================================
def process_video(model, source, is_webcam=False):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    # Tạo video output (nếu không phải webcam)
    out = None
    if not is_webcam:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            f"output_{os.path.basename(source)}", fourcc, fps, (width, height)
        )

    frame_count = 0
    print("🚀 Processing... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Dự đoán mỗi 5 frame (tăng tốc)
        if frame_count % 5 == 0 or frame_count == 1:
            class_name, confidence = predict_frame(model, frame)

        # Vẽ kết quả
        color = (0, 255, 0) if class_name == "Adult" else (0, 0, 255)
        label = f"{class_name}: {confidence*100:.1f}%"
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Hiển thị
        cv2.imshow("Adult/Child Detection", frame)

        # Lưu video output
        if out:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("✅ Done!")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # Configuration
    # SOURCE = 0
    SOURCE = "test/test_video_00001.mp4"
    # SOURCE = "test.jpg"

    # Load model
    model = load_model()

    # Xác định loại source
    if isinstance(SOURCE, int):
        print("🎥 Webcam mode")
        process_video(model, SOURCE, is_webcam=True)
    elif isinstance(SOURCE, str):
        ext = os.path.splitext(SOURCE)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            print("🖼️ Image mode")
            process_image(model, SOURCE)
        else:
            print("🎬 Video mode")
            process_video(model, SOURCE, is_webcam=False)
    else:
        print("❌ Invalid source!")
