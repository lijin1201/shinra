import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import BasicBlock


# ---------------------
# CBAM Modules
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

    def forward(self, x, return_attention=False):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        if return_attention:
            return x * attn, attn
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class CBAMWithMaskSupervision(CBAM):
    def forward(self, x):
        x = self.ca(x)
        x, attn_map = self.sa(x, return_attention=True)
        return x, attn_map


# ---------------------
# Basic Blocks
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


class CBAMBasicBlockWithMask(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAMWithMaskSupervision(self.conv2.out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out, attn_map = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, attn_map


# ---------------------
# Helper Function
# ---------------------
def _replace_block_layer(layer, block_type, return_attention=False):
    new_layers = []
    for orig_block in layer:
        block_cls = block_type
        block = block_cls(orig_block.conv1.in_channels,
                          orig_block.conv2.out_channels,
                          stride=orig_block.stride,
                          downsample=orig_block.downsample)
        new_layers.append(block)
    return nn.Sequential(*new_layers)

def _replace_block_layer_last(layer, block_type, return_attention=False):
    new_layers = []
    for i, orig_block in enumerate(layer):
        if i == len(layer) - 1:
            block_cls = block_type
            block = block_cls(orig_block.conv1.in_channels,
                            orig_block.conv2.out_channels,
                            stride=orig_block.stride,
                            downsample=orig_block.downsample)
            new_layers.append(block)
        else:
            new_layers.append(orig_block)
    return nn.Sequential(*new_layers)

# ---------------------
# Full Model
# ---------------------
class ResNet18CBAMMask(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = _replace_block_layer(base.layer1, CBAMBasicBlock)
        self.layer2 = _replace_block_layer(base.layer2, CBAMBasicBlock)
        self.layer3 = _replace_block_layer_last(base.layer3, CBAMBasicBlockWithMask)
        self.layer4 = base.layer4

        self.avgpool = base.avgpool
        self.fc = base.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # layer3 returns (features, attn_map)
        x, attn_map = self.layer3(x) 

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, attn_map
