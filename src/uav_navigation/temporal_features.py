from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TemporalObstacleFeatureExtractor(BaseFeaturesExtractor):
    """Lightweight static/dynamic encoder for the 8-frame structured observation."""

    def __init__(
        self,
        observation_space,
        history_length: int = 8,
        current_size: int = 54,
        lidar_size: int = 16,
        obstacle_size: int = 18,
    ):
        if history_length < 2:
            raise ValueError("temporal feature extractor requires at least two frames")
        expected_size = current_size + (history_length - 1) * (lidar_size + obstacle_size)
        if observation_space.shape != (expected_size,):
            raise ValueError(
                f"expected observation shape {(expected_size,)}, got {observation_space.shape}"
            )
        super().__init__(observation_space, features_dim=128)
        self.history_length = history_length
        self.current_size = current_size
        self.lidar_size = lidar_size
        self.obstacle_size = obstacle_size
        self.perception_size = lidar_size + obstacle_size

        self.static_encoder = nn.Sequential(
            nn.Linear(current_size, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
        )
        self.lidar_conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=(1, 0))
        self.lidar_conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 0))
        self.lidar_projection = nn.Sequential(
            nn.Linear(16 * lidar_size, 32),
            nn.SiLU(),
        )
        self.obstacle_gru = nn.GRU(obstacle_size, 32, batch_first=True)

    def _split_sequence(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = observations[:, : self.current_size]
        history = observations[:, self.current_size :].reshape(
            -1, self.history_length - 1, self.perception_size
        )
        current_perception = torch.cat(
            (
                current[:, 20 : 20 + self.lidar_size],
                current[:, 20 + self.lidar_size : 20 + self.lidar_size + self.obstacle_size],
            ),
            dim=1,
        ).unsqueeze(1)
        sequence = torch.cat((history, current_perception), dim=1)
        return current, sequence

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        current, sequence = self._split_sequence(observations)
        static_features = self.static_encoder(current)

        lidar = sequence[:, :, : self.lidar_size].unsqueeze(1)
        lidar = functional.pad(lidar, (1, 1, 0, 0), mode="circular")
        lidar = functional.silu(self.lidar_conv1(lidar))
        lidar = functional.pad(lidar, (1, 1, 0, 0), mode="circular")
        lidar = functional.silu(self.lidar_conv2(lidar))
        lidar = lidar.mean(dim=2).flatten(start_dim=1)
        lidar_features = self.lidar_projection(lidar)

        obstacle_sequence = sequence[:, :, self.lidar_size :]
        _, obstacle_hidden = self.obstacle_gru(obstacle_sequence)
        obstacle_features = obstacle_hidden[-1]
        return torch.cat((static_features, lidar_features, obstacle_features), dim=1)
