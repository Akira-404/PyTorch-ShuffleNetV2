from typing import List, Callable
import torch
import torch.nn as nn
from torch import Tensor

'''
input:([N,6,9,9],3)
channels->[c1,c2,c3,c4,c5,c6]

channels_per_group:6//3=2
->[N,3,2,9,9]
-[c1,c2],[c3,c4],[c5,c6]

->[N,2,3,9,9]
-[c1,c3,c5],[c2,c4,c6]

->[N,6,9,9]
-[c1,c3,c5,c2,c4,c6]
'''


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()

    # 每一组含有的通道数
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size,num_channels,height,width]->[batch_size,groups,channels_per_group,height,width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # [batch_size,groups,channels_per_group,height,width]->[batch_size,channels_per_group,groups,height,width]
    x = torch.transpose(x, 1, 2).contiguous()

    # 展开
    x = x.view(batch_size, -1, height, width)

    return x


'''
strid=1:
input->channel split
[1x1 conv]->[3x3 dwconv]->[1x1 conv]->ret
concat(input,ret)->(channel shuffle)->output
------------------------------
------------------------------
strid=2:
input->[3x3 dwconv,s=2]->[1x1 conv]->ret1
input->[1x1 conv]->[3x3 dwconv,s=2]->[1x1 conv]->ret2
concat(ret1,ret2)->(channel suffle)->output
'''


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError('illegal stride value')
        self.stride = stride

        assert output_c % 2 == 0

        # 当stride==1时，分支通道//2
        branch_features = output_c // 2

        # 当stride!=1时，input_c==分支通道x2
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),

            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

    @staticmethod
    def depthwise_conv(input_c: int, output_c: int, kernel_s: int, stride: int = 1, padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s, stride=stride,
                         padding=padding, bias=bias, groups=input_c)


'''
image->output:224x224 out channel:3

conv1:112x112 ksize:3x3,s=2,repeat:1,out channel:24
maxpool:output size 56x56 ksize:3x3 s:2 repeat:1 out channel:24

stage2:output size 28x28  Stride:2 repeat:1 out channel:116
stage2:output size 28x28  Stride:1 repeat:3 out channel:116

stage3:output size 14x14  Stride:2 repeat:1 out channel:232
stage3:output size 14x14  Stride:1 repeat:7 out channel:232

stage4:output size 7x7    Stride:1 repeat:1 out channel:464
stage4:output size 7x7    Stride:1 repeat:3 out channel:464

conv5:output size 7x7 ksize:1x1 Stride:1 repeat:1 out channel:1024
globalpool: output size 7x7 ksize:7x7

fc:output channels:1000

'''


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats: List[int], stages_out_channels: List[int], num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):

            seq = [inverted_residual(input_channels, output_channels, 2)]

            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))

            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.conv5(x)
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# weight:https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
def shufflenet_v2_x1(num_classes=1000):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)
    return model


# weight:https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth
def shufflenet_v2_x0_5(num_classes=1000):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)
    return model
