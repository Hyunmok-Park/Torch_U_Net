import torch
from torch import nn
from torch.nn import Module


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, pool_kernel_size):
        super().__init__()

        self.convlayer1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        self.convlayer2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)

        self.layer = nn.Sequential(
            self.convlayer1,
            self.batchnorm1,
            nn.ReLU(),
            self.convlayer2,
            self.batchnorm2,
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)

    def forward(self, inputs):
        output_for_decoder = self.layer(inputs)
        output_for_next = self.pool(output_for_decoder)
        return output_for_next, output_for_decoder


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, pool_kernel_size):
        super().__init__()

        self.convlayer1 = nn.Conv2d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        self.convlayer2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)

        self.pool = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=pool_kernel_size, stride=2)

        self.layer = nn.Sequential(
            self.convlayer1,
            self.batchnorm1,
            nn.ReLU(),
            self.convlayer2,
            self.batchnorm2,
            nn.ReLU()
        )


    def forward(self, encoder_output, decoder_output):
        inputs = self.pool(decoder_output)
        inputs = torch.cat([encoder_output, inputs], dim=1)
        output = self.layer(inputs)
        return output
