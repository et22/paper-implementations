import torch.nn as nn

class ImageTNet(nn.Module):
    def __init__(self):
        super(ImageTNet, self).__init__()
        # see supplement for network architecture
        pad1 = nn.ReflectionPad2d(40)
        conv1 = ImageTNet._conv(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4))
        down1 = ImageTNet._conv(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))
        down2 = ImageTNet._conv(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))
        res1 = ResBlock(128)
        res2 = ResBlock(128)
        res3 = ResBlock(128)
        res4 = ResBlock(128)
        res5 = ResBlock(128)
        up1 = ImageTNet._conv(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1))
        up2 = ImageTNet._conv(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1))
        conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding = 4)
        tanh = nn.Tanh()
        
        self.model = nn.Sequential(
            pad1,
            conv1,
            down1,
            down2,
            res1,
            res2,
            res3,
            res4,
            res5,
            up1,
            up2,
            conv2,
            tanh,
        )

    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def _conv(conv):
        return nn.Sequential(
            conv, 
            nn.BatchNorm2d(conv.out_channels),
            nn.ReLU(),
        )

class ResBlock(nn.Module):
    def __init__(self, n):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(n)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x[:, :, 2:-2, 2:-2] # center crop x
        return out
