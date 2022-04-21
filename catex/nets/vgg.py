import torch
import torch.nn.functional as F

class VGG2L(torch.nn.Module):
    def __init__(self, in_channel = 4):
        super(VGG2L, self).__init__()
        kernel_size = 3
        padding = 1

        self.conv1_1 = torch.nn.Conv2d(in_channel, 64,
                     kernel_size, stride = 1, padding = padding)

        self.conv2_2 = torch.nn.Conv2d(64, 64,
                    kernel_size, stride = 1,
                    padding = padding)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.conv2_1 = torch.nn.Conv2d(
            64, 128, kernel_size, stride = 1, pading = padding
        )

        self.conv2_2 = torch.nn.Conv2d(
            128, 128, kernel_size, stride = 1, padding = padding
        )

        self.bn2 = torch.nn.BatchNorm(128)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens):
        xs_pad = xs_pad.view(
            xs_pad.size(0), xs_pad.size(1),
            self.in_channel,
            xs_pad.size(2) // self.in_channel
        ).transpose(1, 2)

        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = self.bn1(xs_pad)

        xs_pad = F.max_pool2d(
            xs_pad, [1, 2], stride = [1, 2],
            ceil_mode = True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = self.bn2(xs_pad)

        xs_pad = F.max_pool2d(xs_pad, [1, 2],
         stride = [1, 2], ceil_mode = True)

        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3)
        )
        return xs_pad, ilens