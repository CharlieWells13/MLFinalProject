import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if pretrained:
            self._load_pretrained_backbone()

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _load_pretrained_backbone(self) -> None:
        tv_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        tv_state = tv_model.state_dict()
        own_state = self.state_dict()

        mapped_state: dict[str, torch.Tensor] = {}
        for k, v in tv_state.items():
            if k.startswith("fc."):
                continue
            if k == "conv1.weight":
                mapped_key = "stem.0.weight"
            elif k.startswith("bn1."):
                mapped_key = "stem.1." + k.split("bn1.", 1)[1]
            else:
                mapped_key = k
            if mapped_key in own_state and own_state[mapped_key].shape == v.shape:
                mapped_state[mapped_key] = v

        self.load_state_dict(mapped_state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet18BBoxRegressor(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2, apply_sigmoid: bool = False):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        preds = self.head(features)
        if self.apply_sigmoid:
            preds = torch.sigmoid(preds)
        return preds


def build_model(pretrained: bool = True, freeze_backbone: bool = False, apply_sigmoid: bool = False) -> ResNet18BBoxRegressor:
    model = ResNet18BBoxRegressor(pretrained=pretrained, apply_sigmoid=apply_sigmoid)
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    return model
