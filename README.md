# Adult & Child Detection

Phân loại người lớn / trẻ em từ ảnh, video hoặc webcam bằng ResNet-18 tự xây dựng.

![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/pytorch-latest-ee4c2c?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)


## Tính năng

- Phân loại nhị phân: Adult / Child với confidence score
- Suy luận real-time trên webcam, video file, hoặc ảnh tĩnh
- Kiến trúc ResNet-18 tự implement từ đầu (không dùng pretrained)
- Augmentation mạnh: ColorJitter, RandomErasing, RandomCrop
- Giao diện web upload ảnh / chụp camera qua Streamlit


## Tech Stack

Python · PyTorch · torchvision · OpenCV · Streamlit · NumPy · Pillow · scikit-learn · TensorBoard · tqdm · matplotlib


## Bắt đầu


### Yêu cầu

- Python 3.8+
- CUDA (tùy chọn, hỗ trợ CPU fallback tự động)


### Cài đặt

```bash
git clone https://github.com/<your-username>/detect_adult-child_resnet.git
cd detect_adult-child_resnet
pip install -r requirement.txt
```


### Cấu trúc dataset

```
datasets/
├── train/
│   ├── Adult/
│   └── Child/
├── val/
│   ├── Adult/
│   └── Child/
└── test/
    ├── Adult/
    └── Child/
```


### Huấn luyện

```bash
python train.py
```

Model tốt nhất được lưu tại `best_resnet_model.pth`. Biểu đồ loss/accuracy lưu tại `training_curves.png`.

Các siêu tham số mặc định:

| Tham số | Giá trị |
|---|---|
| Epochs | 60 |
| Batch size | 64 |
| Learning rate | 0.0001 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Input size | 224×224 |


## Sử dụng


### Inference (CLI)

Mở `main.py` và chỉnh biến `SOURCE`:

```python
SOURCE = 0              # Webcam
SOURCE = "test/video.mp4"   # Video file
SOURCE = "test/image.jpg"   # Ảnh tĩnh
```

```bash
python main.py
```

Nhấn `q` để thoát khi xử lý video / webcam.


### Giao diện web (Streamlit)

```bash
streamlit run ui.py
```

Truy cập `http://localhost:8501` — upload ảnh (jpg/png) hoặc video mp4, hoặc chụp trực tiếp từ camera.


## Cấu hình

| Biến | Mặc định | Mô tả |
|---|---|---|
| `MODEL_PATH` | `best_resnet_model.pth` | Đường dẫn đến model đã train |
| `CLASS_NAMES` | `["Adult", "Child"]` | Nhãn phân loại |
| `DEVICE` | auto (cuda/cpu) | Thiết bị suy luận |
| `DISPLAY_WIDTH` | `800` | Chiều rộng hiển thị cửa sổ |
| `DISPLAY_HEIGHT` | `600` | Chiều cao hiển thị cửa sổ |
| `data_dir` | `datasets` | Thư mục dataset khi train |


## Cấu trúc dự án

```
detect_adult-child_resnet/
├── models/
│   └── model.py          # ResNet-18 & ResidualBlock
├── utils/
│   ├── resize_image.py   # Tiền xử lý ảnh
│   └── manual_label.py   # Gán nhãn thủ công
├── datasets/             # Dữ liệu train/val/test
├── test/                 # Ảnh/video test
├── train.py              # Training loop
├── main.py               # Inference CLI
├── ui.py                 # Giao diện Streamlit
├── best_resnet_model.pth # Model đã train
└── requirement.txt       # Dependencies
```


## Đóng góp

Pull request và issue luôn được chào đón. Fork repo, tạo branch mới, commit rõ ràng, rồi mở PR. Ưu tiên các cải tiến về độ chính xác, tốc độ inference, hoặc mở rộng thêm lớp phân loại.


## License

MIT