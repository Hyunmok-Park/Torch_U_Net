import torch
from torch import nn
from torch.nn import Module

from net.block import EncoderBlock
from net.block import DecoderBlock


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.e_block1 = EncoderBlock(1, 64, 3, 1, 0, True, 2)
        self.e_block2 = EncoderBlock(64, 128, 3, 1, 0, True, 2)
        self.e_block3 = EncoderBlock(128, 256, 3, 1, 0, True, 2)
        self.e_block4 = EncoderBlock(256, 512, 3, 1, 0, True, 2)

        self.bridge = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.d_block1 = DecoderBlock(1024, 512, 3, 1, 0, True, 2)
        self.d_block2 = DecoderBlock(512, 256, 3, 1, 0, True, 2)
        self.d_block3 = DecoderBlock(256, 128, 3, 1, 0, True, 2)
        self.d_block4 = DecoderBlock(128, 64, 3, 1, 0, True, 2)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=1)

    def forward(self, inputs):
        e_output1, d_inputs1 = self.e_block1(inputs)
        e_output2, d_inputs2 = self.e_block1(e_output1)
        e_output3, d_inputs3 = self.e_block1(e_output2)
        e_output4, d_inputs4 = self.e_block1(e_output3)

        e_output = self.bridge(e_output4)

        d_output1 = self.d_block1(e_output, d_inputs4)
        d_output2 = self.d_block1(d_output1, d_inputs3)
        d_output3 = self.d_block1(d_output2, d_inputs2)
        d_output4 = self.d_block1(d_output3, d_inputs1)

        output = self.final_conv(d_output4)

        return output