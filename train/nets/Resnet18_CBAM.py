import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import BasicBlock


# ---------------------
# CBAM MODULE
# ---------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.shared_mlp(self.avg_pool(x).view(b, c))
        max_out = self.shared_mlp(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        scale = self.sigmoid(out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(x_cat))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ---------------------
# Modified BasicBlock with CBAM
# ---------------------
class CBAMBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAM(self.conv2.out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ---------------------
# Load Pretrained ResNet18 with CBAM Blocks
# ---------------------
def resnet18_cbam(pretrained=True,**kwargs):
    # Load original ResNet18
    original_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,**kwargs)

    # Create new model using CBAMBasicBlock
    model = resnet18(weights=None)
    model.layer1 = _replace_block_layer(original_resnet.layer1, CBAMBasicBlock)
    model.layer2 = _replace_block_layer(original_resnet.layer2, CBAMBasicBlock)
    model.layer3 = _replace_block_layer(original_resnet.layer3, CBAMBasicBlock)
    # model.layer4 = _replace_block_layer(original_resnet.\, CBAMBasicBlock)

    # Load original weights where possible
    if pretrained:
        model.load_state_dict(original_resnet.state_dict(), strict=False)

    return model


# ---------------------
# Helper: Replace Blocks with CBAM Versions
# ---------------------
def _replace_block_layer(layer, block_type):
    new_layers = []
    for orig_block in layer:
        cbam_block = block_type(orig_block.conv1.in_channels,
                                orig_block.conv2.out_channels,
                                stride=orig_block.stride,
                                downsample=orig_block.downsample)
        new_layers.append(cbam_block)
    return nn.Sequential(*new_layers)
