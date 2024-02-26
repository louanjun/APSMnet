import torch
from torch import nn
import torch.nn.functional as F

'''Implementation for CONVOLUTIONAL NEURAL NETWORKS FOR HYPERSPECTRAL IMAGE CLASSIFICATION'''

class CNN_HSI(nn.Module):
    def __init__(self, in_channel, nc):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 128, 1),
            nn.ReLU(),
            nn.LocalResponseNorm(3),
            nn.Dropout(0.6)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.LocalResponseNorm(3),
            nn.Dropout(0.6)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, nc, 1),
            nn.ReLU(),
            nn.AvgPool2d(5, 1)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = out3.view(out3.size(0),-1)
        return out4


#net = CNN_HSI(204, 16)
#input = torch.rand((2,204,5,5))
#out = net(input)
#print(out.shape)


