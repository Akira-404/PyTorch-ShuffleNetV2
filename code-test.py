import torch
import torch.nn as nn
from torch import Tensor


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    print("batch_size:{}, num_channels:{}, height:{}, width:{}".format(batch_size, num_channels, height, width))

    # 每一组含有的通道数
    channels_per_group = num_channels // groups
    print('每一组含有的通道数:{}'.format(channels_per_group))

    # reshape
    # [batch_size,num_channels,height,width]->[batch_size,groups,channels_per_group,height,width]
    x = x.view(batch_size, groups, channels_per_group, height, width)
    print(x.size())
    print(x)
    # [batch_size,groups,channels_per_group,height,width]->[batch_size,channels_per_group,groups,height,width]
    x = torch.transpose(x, 1, 2).contiguous()
    print(x.size())
    print(x)
    # 展开
    x = x.view(batch_size, -1, height, width)

    return x


if __name__ == '__main__':
    img1 = torch.tensor([[[[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]],

                          [[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 2]],

                          [[3, 3, 3],
                           [3, 3, 3],
                           [3, 3, 3]],

                          [[4, 4, 4],
                           [4, 4, 4],
                           [4, 4, 4]],

                          [[5, 5, 5],
                           [5, 5, 5],
                           [5, 5, 5]],

                          [[6, 6, 6],
                           [6, 6, 6],
                           [6, 6, 6]]]])
    print("input szie:", img1.size())
    ret = channel_shuffle(img1, 2)
    print('ret size:', ret.size())
    print('ret:', ret)

    # conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, groups=2)
    # print("groups=2", conv.weight.data.size())
    # conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, groups=3)
    # print("groups=3", conv.weight.data.size())
    # conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, groups=6)
    # print("groups=6", conv.weight.data.size())
