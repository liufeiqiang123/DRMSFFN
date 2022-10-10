import torch
import torch.nn as nn
from .blocks import MeanShift

#Multi-scale-feature-fusion-block(MSFFB)
class MSFFB(nn.Module):
    def __init__(self, num_features):
        super(MSFFB, self).__init__()

        self.conv1 = nn.Sequential(*[
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, stride=1),
            nn.PReLU()
        ])

        self.conv2 = nn.Sequential(*[
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, stride=1),
            nn.PReLU()
        ])

        self.conv3 = nn.Sequential(*[
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, stride=1),
            nn.PReLU()
        ])

        self.conv4 = nn.Sequential(*[
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, stride=1),
            nn.PReLU()
        ])

        self.conv5 = nn.Sequential(*[
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, stride=1),
            nn.PReLU()
        ])

        self.conv6 = nn.Sequential(*[
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, stride=1),
            nn.PReLU()
        ])

        self.conv7 = nn.Sequential(*[
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, stride=1),
            nn.PReLU()
        ])

        self.conv8 = nn.Sequential(*[
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, stride=1),
            nn.PReLU()
        ])

        self.cat1 = nn.Sequential(*[
            nn.Conv2d(in_channels=2*num_features, out_channels=num_features, kernel_size=1, padding=0, stride=1),
            nn.PReLU()
        ])

        self.cat2 = nn.Sequential(*[
            nn.Conv2d(in_channels=2*num_features, out_channels=num_features, kernel_size=1, padding=0, stride=1),
            nn.PReLU()
        ])

        self.cat3 = nn.Sequential(*[
            nn.Conv2d(in_channels=2*num_features, out_channels=num_features, kernel_size=1, padding=0, stride=1),
            nn.PReLU()
        ])

        self.cat4 = nn.Sequential(*[
            nn.Conv2d(in_channels=2*num_features, out_channels=num_features, kernel_size=1, padding=0, stride=1),
            nn.PReLU()
        ])

        self.cat5 = nn.Sequential(*[
            nn.Conv2d(in_channels=2*num_features, out_channels=num_features, kernel_size=1, padding=0, stride=1),
            nn.PReLU()
        ])

        self.cat6 = nn.Sequential(*[
            nn.Conv2d(in_channels=2*num_features, out_channels=num_features, kernel_size=1, padding=0, stride=1),
            nn.PReLU()
        ])

        self.cat7 = nn.Sequential(*[
            nn.Conv2d(in_channels=2*num_features, out_channels=num_features, kernel_size=1, padding=0, stride=1),
            nn.PReLU()
        ])

        #Multi-scale feature fusion unit (MFF)
        self.MFF = nn.Sequential(*[
            nn.Conv2d(in_channels=8 * num_features, out_channels=num_features, kernel_size=1, padding=0, stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, stride=1),
            nn.PReLU()
        ])

    def forward(self, x):
        x1 = self.conv1(x)
        x11 = self.cat1(torch.cat((x, x1),dim=1))

        x2 = self.conv2(x11)
        x22 = self.cat2(torch.cat((x, x2),dim=1))

        x3 = self.conv3(x22)
        x33 = self.cat3(torch.cat((x, x3),dim=1))

        x4 = self.conv4(x33)
        x44 = self.cat4(torch.cat((x, x4),dim=1))

        x5 = self.conv5(x44)
        x55 = self.cat5(torch.cat((x, x5),dim=1))

        x6 = self.conv6(x55)
        x66 = self.cat6(torch.cat((x, x6),dim=1))

        x7 = self.conv7(x66)
        x77 = self.cat7(torch.cat((x, x7),dim=1))

        x8 = self.conv8(x77)

        x9 = self.MFF(torch.cat((x1, x2, x3, x4, x5, x6, x7, x8),dim=1))

        out = x + x9
        return out

#Deep recursive multi-scale feature fusion network (DRMSFFN)
class DRMSFFN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_recursion, upscale_factor):
        super(DRMSFFN, self).__init__()
        r = upscale_factor
        if r == 2:
            S = 2
            P = 2
            K = 6
        elif r == 3:
            S = 3
            P = 2
            K = 7
        elif r == 4:
            S = 4
            P = 2
            K = 8
        elif r == 8:
            S = 8
            P = 2
            K = 12
        else:
            raise ValueError("scale must be 2 or 3 or 4 or 8.")

        self.upscale_factor = upscale_factor

        self.D = num_recursion

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # initial low-level feature extraction block
        self.head = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, stride=1),
            nn.PReLU()
        ])

        #Multi-Scale feature-fusion-block
        self.block = MSFFB(num_features)

        # Global Feature Fusion
        self.conv1s = nn.ModuleList()
        for i in range(self.D):
            self.conv1s.append(
                nn.Conv2d(in_channels=2*num_features, out_channels=num_features, kernel_size=1, padding=0, stride=1)
            )

        self.UPNet = nn.Sequential(*[
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, stride=1),
            nn.ConvTranspose2d(in_channels=num_features, out_channels=num_features,kernel_size=K,stride=S,padding=P),
            nn.PReLU(),
            nn.Conv2d(in_channels=num_features,out_channels=out_channels,kernel_size=3, padding=1, stride=1)
        ])

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)

        fea = self.head(x)
        f1 = fea

        for i in range(self.D):
            f1 = self.block(f1)
            f1 = self.conv1s[i](torch.cat((fea, f1),dim=1))

        out = self.UPNet(f1) + inter_res
        h = self.add_mean(out)
        return h

