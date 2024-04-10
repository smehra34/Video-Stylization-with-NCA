'''
Modified from base implementation in:
    Sudhakaran, S., Najarro, E., & Risi, S. (2022).
    "Goal-guided neural cellular automata: Learning to control self-organising systems."

https://github.com/shyamsn97/controllable-ncas/blob/master/controllable_nca/nca.py
'''

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding

from encoder import ImageEncoder

# class Encoder(nn.Module):
#     def __init__(self, num_embeddings: int, out_channels: int):
#         super().__init__()
#         self.num_embeddings = num_embeddings
#         self.embedding = Embedding(num_embeddings, out_channels)
#
#     def forward(self, indices):
#         embeddings = self.embedding(indices)
#         return embeddings


class UpdateNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        zero_bias: bool = True,
    ):

        super(UpdateNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, 64, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, self.out_channels, 1, bias=False),
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        torch.nn.init.zeros_(m.bias)

        with torch.no_grad():
            self.apply(init_weights)

    def forward(self, x):
        return self.out(x)


class ConditionedNCA(torch.nn.Module):
    def __init__(
        self,
        #num_goals: int,
        encoder: nn.Module = None,
        # use_image_encoder: bool = False,
        target_shape: Tuple[int] = (3, 64, 64),
        num_hidden_channels=16,
        use_living_channel: bool = True,
        living_channel_dim: Optional[int] = None,
        alpha_living_threshold: float = 0.1,
        cell_fire_rate: float = 0.5,
        zero_bias=True,
    ):
        super().__init__()
        # self.num_goals = num_goals

        self.target_shape = target_shape
        self.num_target_channels = self.target_shape[0]

        self.image_size = self.target_shape[-1]  # height of image

        self.num_hidden_channels = num_hidden_channels

        self.use_living_channel = use_living_channel
        self.living_channel_dim = living_channel_dim
        if self.living_channel_dim is None:
            self.living_channel_dim = self.num_target_channels

        self.num_channels = (
            self.num_target_channels + self.num_hidden_channels + 1
        )  # target_channels + hidden channels + living_channel_dim

        self.alpha_living_threshold = alpha_living_threshold
        self.cell_fire_rate = cell_fire_rate
        self.zero_bias = zero_bias

        # setup network
        self.perception_net = torch.nn.Conv2d(
            self.num_channels,
            self.num_channels * 3,
            3,
            stride=1,
            padding=1,
            groups=self.num_channels,
            bias=False,
        )
        self.update_net = UpdateNet(
            self.num_channels * 3, self.num_channels, self.zero_bias
        )
        # self.use_image_encoder = use_image_encoder
        self.encoder = encoder
        if encoder is None:
            self.encoder = ImageEncoder(self.num_hidden_channels, self.num_target_channels)
        # if encoder is None:
        #     if self.use_image_encoder:
        #         self.encoder = build_conv2d_net(
        #             target_shape[0],
        #             self.num_hidden_channels,
        #             target_shape[-1],
        #             1,
        #             hidden_dims=64,
        #         )
        #     else:
        #         self.encoder = Encoder(self.num_goals, self.num_hidden_channels)

    def encode(self, images: torch.Tensor):
        return self.encoder(images)

    def generate_seed(
        self,
        num_seeds,
        device: Optional[torch.device] = None,
        size: Optional[int] = None,
    ):
        if device is not None:
            device = torch.device("cpu")
        if size is None:
            size = self.image_size
        seed = torch.zeros(
            num_seeds,
            self.num_channels,
            size,
            size,
            device=device,
        )
        seed[
            :, self.living_channel_dim :, size // 2, size // 2
        ] = 1.0  # rgb=0, alpha=1 = black
        return seed

    def alive(self, x):
        if not self.use_living_channel:
            return torch.ones_like(x, dtype=torch.bool, device=x.device)
        return (
            F.max_pool2d(
                x[:, self.living_channel_dim : self.living_channel_dim + 1, :, :],
                kernel_size=3,
                stride=1,
                padding=1,
            )
            > self.alpha_living_threshold
        )

    def get_stochastic_update_mask(self, x):
        """
        Return stochastic update mask
        Args:
            x ([type]): [description]
        """
        return (
            torch.clamp(torch.rand_like(x[:, 0:1], device=x.device), 0.0, 1.0).float()
            < self.cell_fire_rate
        ).float()

    def update(self, x, goal_encoding, pre_life_mask):
        perceive = self.perception_net(x + goal_encoding * pre_life_mask)
        out = self.update_net(perceive)
        return out

    def forward(self, x):

        x, goal_encoding = x[0], x[1]

        pre_life_mask = self.alive(x)

        rand_mask = self.get_stochastic_update_mask(x)
        out = self.update(x, goal_encoding, pre_life_mask)
        x = x + rand_mask * out

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        x = torch.clamp(x, -10.0, 10.0)
        return x, goal_encoding

    def grow(self, x: torch.Tensor, num_steps: int, goal: torch.Tensor) -> torch.Tensor:
        goal_encoding = self.encoder(goal)#.view(x.size(0), -1)
        if goal_encoding.size(1) == self.num_hidden_channels:
            goal_encoding = F.pad(
                goal_encoding,
                (0, 0, 0, 0, self.num_channels - self.num_hidden_channels, 0),
            )  # pad initial channels with zeros
        # goal_encoding = goal_encoding.view(x.size(0), self.num_channels, 1, 1).repeat(
        #     1, 1, x.size(-1), x.size(-1)
        # )
        for _ in range(num_steps):
            x, goal_encoding = self.forward((x, goal_encoding))
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
