import torch
import torch.nn as nn
import numpy as np

from controllable_nca.utils import build_conv2d_net

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim, channels):
        super(ImageEncoder, self).__init__()

        self.channels = channels

        # sobel filters
        sobel_x_weight = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y_weight = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x.weight = nn.Parameter(sobel_x_weight, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_weight, requires_grad=False)

        # gaussian blur filter
        gaussian_kernel = self._gaussian_kernel(5, 1)
        self.gaussian_blur = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        self.gaussian_blur.weight = nn.Parameter(gaussian_kernel, requires_grad=False)

        # laplacian filter
        laplacian_weight = torch.tensor([[[[1, 2, 1], [2, -12, 2], [1, 2, 1]]]], dtype=torch.float32)
        self.laplacian = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.laplacian.weight = nn.Parameter(laplacian_weight, requires_grad=False)

        # convolutions to get embeddings
        self.embed = nn.Sequential(
            nn.Conv2d(channels + 3, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1, bias=False)
        )


    def forward(self, x):

        # convert input to grayscale (for sobel and laplacian filters)
        x_gray = torch.mean(x, dim=1, keepdim=True)

        sobel_x_out = self.sobel_x(x_gray)
        sobel_y_out = self.sobel_y(x_gray)
        laplacian_out = self.laplacian(x_gray)

        # apply channelwise blurring
        blurred_channels = [self.gaussian_blur(x[:, i:i+1]) for i in range(self.channels)]
        blurred_out = torch.cat(blurred_channels, dim=1)

        # stack the transformations
        output = torch.cat((sobel_x_out, sobel_y_out, laplacian_out, blurred_out), dim=1)

        # print(f"{sobel_x_out.shape=}, {sobel_y_out.shape=}, {laplacian_out.shape=}, {blurred_out.shape=}, {output.shape=}")

        # applying convolution to get pixelwise embeddings
        output = self.embed(output)
        return output


    def _gaussian_kernel(self, size, sigma=1):
        kernel = torch.tensor([[(1/(2 * np.pi * sigma**2)) * np.exp(-((i - size//2)**2 + (j - size//2)**2) / (2 * sigma**2)) for j in range(size)] for i in range(size)])
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, size, size)
        return kernel.float()
