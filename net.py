import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import time
import timm
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from depth import DepthBranch
from mobilenet import MobileNetV2Encoder
from RGB_net import RGBBranch

def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)

class DFMNet(nn.Module):
    def __init__(self, **kwargs):
        super(DFMNet, self).__init__()
        self.rgb = RGBBranch()
        self.depth = DepthBranch()

    def forward(self, r, d):
        size = r.shape[2:]
        outputs = []

        sal_d,feat = self.depth(r,d)
        sal_final= self.rgb(r,feat)

        sal_final = upsample(sal_final, size)
        sal_d = upsample(sal_d, size)

        # sal_d = sal_d.mean(dim=1)  # 或者使用 sal_d.sum(dim=1)
        # sal_d = sal_d.unsqueeze(1)  # 添加一个维度，形状变为 [10, 1, 256, 256]

        outputs.append(sal_final)
        outputs.append(sal_d)

        return outputs

if __name__ == '__main__':
    img = torch.randn(1, 3, 256, 256).cuda()
    depth = torch.randn(1, 1, 256, 256).cuda()
    model = DFMNet().cuda()
    model.eval()
    time1= time.time()
    outputs = model(img,depth)
    time2 = time.time()
    torch.cuda.synchronize()
    print(1000/(time2-time1))
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(num_params)
