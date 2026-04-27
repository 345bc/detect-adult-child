"""
ResNet-18 model for Adult/Child classification
Input: 224x224x3 images
Output: 2 classes (Adult, Child)
"""

import torch
import torch.nn as nn


# ResidualBlock solves the vanishing gradient problem using skip connection
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)

        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()

        self.in_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    print("=" * 50)
    print("KIỂM TRA MÔ HÌNH RESNET-18")
    print("=" * 50)

    # 1. Tạo model
    print("\n1. Đang tạo model...")
    model = ResNet18(num_classes=2)

    # 2. Tạo dữ liệu giả (batch=4, 3 kênh màu, 224x224)
    print("2. Tạo dữ liệu đầu vào giả...")
    dummy_input = torch.randn(4, 3, 224, 224)
    print(f"   Input shape: {dummy_input.shape}")

    # 3. Chạy forward
    print("3. Chạy forward qua mạng...")
    output = model(dummy_input)
    print(f"   Output shape: {output.shape}")

    # 4. Kiểm tra kết quả
    if output.shape == (4, 2):
        print("\n✅ ĐÚNG: Output shape là (4, 2)")
    else:
        print(f"\n❌ SAI: Output shape phải là (4, 2), nhưng được {output.shape}")

    # 5. Đếm số tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 THÔNG SỐ MÔ HÌNH:")
    print(f"   Tổng số tham số: {total_params:,}")
    print(f"   Tham số train được: {trainable_params:,}")
    print(f"   Kích thước model: {total_params * 4 / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 50)
    print("🎉 MÔ HÌNH HOẠT ĐỘNG BÌNH THƯỜNG!")
    print("=" * 50)
