"""独立 CNN 定义：4 个卷积层、2 个全连接层，输出未归一化 logits。"""
import torch
from torch import nn


class CancerCNN(nn.Module):
    """输入 [N,3,H,W]（H,W >= 16），输出 [N,num_classes]。

    四个卷积块通道数为 32/64/128/256，每块包含 3x3 卷积、
    BatchNorm、ReLU、2x2 最大池化；全局池化后使用 128 单元隐藏层。
    CrossEntropyLoss 内部处理 softmax，训练前不要自行转换概率。
    """
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        channels = 3
        for width in (32, 64, 128, 256):
            layers += [nn.Conv2d(channels, width, 3, padding=1, bias=False),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 128),
                                        nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(num_classes: int, config: dict, initialize_pretrained=False) -> nn.Module:
    """旧配置默认使用 CancerCNN，保证已保存权重仍可加载。"""
    architecture = config.get('architecture', 'cancer_cnn')
    dropout = config.get('dropout', 0.3)
    if architecture == 'cancer_cnn':
        return CancerCNN(num_classes, dropout)
    if architecture == 'resnet18':
        from torchvision.models import resnet18, ResNet18_Weights

        pretrained = config.get('pretrained', False)
        if not isinstance(pretrained, bool):
            raise ValueError('pretrained must be a boolean')
        # Only a new training run downloads initial weights; checkpoint loading stays offline.
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained and initialize_pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
        return model
    raise ValueError(f'Unknown model architecture: {architecture}')
