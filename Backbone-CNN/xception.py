import torch
import torch.nn as nn

class depthwise_separable(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, padding, bias=False):
        super(depthwise_separable, self).__init__()
        self.depthwise = nn.Conv2d(n_in, n_in, kernel_size=kernel_size, padding=padding,groups=n_in, bias=bias)
        self.pointwise = nn.Conv2d(n_in, n_out, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Xception(nn.Module):
    def __init__(self, in_channel=3, num_classes=10):
        super(Xception, self).__init__()

        # Entry Flow
        self.entry_basic = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.entry_flow1 = nn.Sequential(
            depthwise_separable(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            depthwise_separable(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow1_residual = nn.Conv2d(64, 128, kernel_size=1, stride=2)

        self.entry_flow2 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            depthwise_separable(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow2_residual = nn.Conv2d(128, 256, kernel_size=1, stride=2)

        self.entry_flow3 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable(256, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(True),
            depthwise_separable(728, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow3_residual = nn.Conv2d(256, 728, kernel_size=1, stride=2)

        # Middle Flow
        self.middle_flow = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable(728, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            depthwise_separable(728, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            depthwise_separable(728, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),
        )
        
        # Exit Flow
        self.exit_flow_1 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable(728, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            depthwise_separable(728, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.exit_flow_1_residual = nn.Conv2d(728, 1024, kernel_size=1, stride=2)
        self.exit_flow_2 = nn.Sequential(
            depthwise_separable(1024, 1536, 3, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(True),
            
            depthwise_separable(1536, 2048, 3, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        entry_out0 = self.entry_basic(x)
        entry_out1 = self.entry_flow1(entry_out0) + self.entry_flow1_residual(entry_out0)
        entry_out2 = self.entry_flow2(entry_out1) + self.entry_flow2_residual(entry_out1)
        entry_out3 = self.entry_flow3(entry_out2) + self.entry_flow3_residual(entry_out2)

        middle_out = self.middle_flow(entry_out3) + entry_out3
        for i in range(7): #repeated 8 times
            middle_out = self.middle_flow(middle_out) + middle_out

        exit_out1 = self.exit_flow_1(middle_out) + self.exit_flow_1_residual(middle_out)
        exit_out2 = self.exit_flow_2(exit_out1)

        exit_avg_pool = self.avg_pool(exit_out2)
        exit_avg_pool_flat = exit_avg_pool.view(exit_avg_pool.size(0), -1)

        output = self.fc(exit_avg_pool_flat)

        return output

def test():
    x = torch.rand(size=(4, 3, 299, 299))
    net = Xception()
    Y = net(x)
    print(net)
    print(Y.size())

test()
