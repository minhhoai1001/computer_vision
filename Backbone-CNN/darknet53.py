import torch
import torch.nn as nn

def conv_bn_leaky(in_c, out_c, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU()
    )

#  Residial block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_c):
        super(DarkResidualBlock, self).__init__()
        reduced_channels = int(in_c/2)

        self.layer1 = conv_bn_leaky(in_c, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_bn_leaky(reduced_channels, in_c)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

class Darknet53(nn.Module):
    def __init__(self, block, num_classes) -> None:
        super(Darknet53, self).__init__()
        self.num_classes = num_classes
        self.conv1 = conv_bn_leaky(3, 32)
        self.conv2 = conv_bn_leaky(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_bn_leaky(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_bn_leaky(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_bn_leaky(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_bn_leaky(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

def darknet53(num_classes=10):
    return Darknet53(DarkResidualBlock, num_classes)


# def test():
#     x = torch.rand(size=(4, 3, 224, 224))
#     net = darknet53(num_classes=10)
#     Y = net(x)
#     print(net)
#     print(Y.size())

# test()