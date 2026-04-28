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

# Kích thước hiển thị tối đa
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# Transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ============================================
# LOAD MODEL
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
# DỰ ĐOÁN
# ============================================
def predict_frame(model, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    return CLASS_NAMES[pred.item()], confidence.item()


# ============================================
# XỬ LÝ ẢNH
# ============================================
def process_image(model, image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Cannot read: {image_path}")
        return

    h, w = frame.shape[:2]
    print(f"📐 Original image size: {w}x{h}")

    class_name, confidence = predict_frame(model, frame)
    color = (0, 255, 0) if class_name == "Adult" else (0, 0, 255)
    label = f"{class_name}: {confidence*100:.1f}%"
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    print(f"📊 {os.path.basename(image_path)} → {label}")

    # Resize để hiển thị vừa màn hình
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    cv2.imshow("Result", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================
# XỬ LÝ VIDEO
# ============================================
def process_video(model, source, is_webcam=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    # Lấy kích thước gốc
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📐 Original video size: {width}x{height}, FPS: {fps}")

    # Tính tỉ lệ resize để hiển thị
    scale = min(DISPLAY_WIDTH / width, DISPLAY_HEIGHT / height)
    display_w = int(width * scale)
    display_h = int(height * scale)

    # Tạo video output
    out = None
    if not is_webcam:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            f"output_{os.path.basename(source)}", fourcc, fps, (width, height)
        )
        print(f"🎬 Saving to: output_{os.path.basename(source)}")

    frame_count = 0
    print("🚀 Processing... Press 'q' to quit")
    print(f"📺 Display size: {display_w}x{display_h}")

    # Tạo cửa sổ
    cv2.namedWindow("Adult/Child Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Adult/Child Detection", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 == 0 or frame_count == 1:
            class_name, confidence = predict_frame(model, frame)

        # Vẽ lên frame gốc
        color = (0, 255, 0) if class_name == "Adult" else (0, 0, 255)
        label = f"{class_name}: {confidence*100:.1f}%"
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Resize để hiển thị
        display_frame = cv2.resize(frame, (display_w, display_h))
        cv2.imshow("Adult/Child Detection", display_frame)

        # Lưu video gốc
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
    # ===== CẤU HÌNH =====
    SOURCE = 0  # Webcam
    # SOURCE = "test/video.mp4"                     # Video
    # SOURCE = "test/image.jpg"  # Ảnh

    model = load_model()

    if isinstance(SOURCE, int):
        print("🎥 Webcam mode")
        process_video(model, SOURCE, is_webcam=True)
    elif isinstance(SOURCE, str):
        ext = os.path.splitext(SOURCE)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            print("🖼️ Image mode")
            process_image(model, SOURCE)
        else:
            print("🎬 Video mode")
            process_video(model, SOURCE, is_webcam=False)
    else:
        print("❌ Invalid source!")
