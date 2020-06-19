from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class Scene(nn.Module):
    """Module for scene predictions
    """
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.channels = channels
        self.bn_out = nn.BatchNorm2d(2048, momentum=0.01, track_running_stats=True)

        self.scene_in = nn.Conv2d(in_channels=512, out_channels=self.channels, kernel_size=1)
        self.scene_out = nn.Conv2d(in_channels=self.channels, out_channels=2048, kernel_size=1)

        # Added test conv layers
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256, momentum=0.01, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.01, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(512, momentum=0.01, track_running_stats=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shape:
            - X: :math:`(N, C_in, H, W)` where :math:`C_in = 512`
            - Output: :math:`(N, C_out, H, W)` where :math:`C_out = 2048`
            - Scene pred: :math:`(N, num_scenes)` where :math:`num_scenes = 27`

        """
        # Added conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.scene_in(x)

        pred = torch.mean(torch.flatten(x, start_dim=2), dim=-1)
        pred = nn.LogSoftmax(dim=-1)(pred)

        x = self.scene_out(x)
        x = self.bn_out(x)
        x = F.relu(x)

        return x, pred


class Depth(nn.Module):
    """ Module for depth prediction
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.channels = channels
        self.bn_out = nn.BatchNorm2d(2048, momentum=0.01, track_running_stats=True)

        self.depth_in = nn.Conv2d(in_channels=512, out_channels=self.channels, kernel_size=1)
        self.depth_out = nn.Conv2d(in_channels=self.channels, out_channels=2048, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256, momentum=0.01, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.01, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(512, momentum=0.01, track_running_stats=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shape:
            - X: :math:`(N, C_in, H, W)` where :math:`C_in = 512`
            - Output: :math:`(N, C_out, H, W)` where :math:`C_out = 2048`
            - Depth pred: :math:`(N, 1, H, W)`
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.depth_in(x)

        # Shape is (N, C, H, W)
        pred = torch.mean(x, dim=1, keepdim=True)

        x = self.depth_out(x)
        x = self.bn_out(x)
        x = F.relu(x)

        return x, pred


class Normals(nn.Module):
    """Module for normal predictions
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.channels = channels
        self.bn_out = nn.BatchNorm2d(2048, momentum=0.01, track_running_stats=True)

        self.normals_in = nn.Conv2d(in_channels=512, out_channels=self.channels, kernel_size=1)
        self.normals_out = nn.Conv2d(in_channels=self.channels, out_channels=2048, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256, momentum=0.01, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.01, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(512, momentum=0.01, track_running_stats=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shape:
            - X: :math:`(N, C_in, H, W)` where :math:`C_in = 512`
            - Output: :math:`(N, C_out, H, W)` where :math:`C_out = 2048`
            - Normals pred: :math:`(N, 3, H, W)`
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.normals_in(x)

        normals_x, normals_y, normals_z = torch.split(x, round(self.channels / 3), dim=1)
        normals_x = torch.mean(normals_x, dim=1)
        normals_y = torch.mean(normals_y, dim=1)
        normals_z = torch.mean(normals_z, dim=1)
        pred = torch.stack([normals_x, normals_y, normals_z], dim=1)

        x = self.normals_out(x)
        x = self.bn_out(x)
        x = F.relu(x)

        return x, pred


class ROCK(nn.Module):
    """ ROCK block combining scene, depth and normals prediction
    """

    def __init__(self,
                 aux_tasks: Tuple[str] = ('scene', 'depth', 'normals')) -> None:
        super().__init__()

        self.scene = 'scene' in aux_tasks
        self.depth = 'depth' in aux_tasks
        self.normals = 'normals' in aux_tasks

        self.backbone_channels = 2048
        self.scene_channels = 27
        self.depth_channels = 128
        self.normals_channels = 3 * 128

        self.conv1 = nn.Conv2d(in_channels=self.backbone_channels, out_channels=512, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(512, momentum=0.01, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(512, momentum=0.01, track_running_stats=True)
        self.bn3 = nn.BatchNorm2d(self.backbone_channels, momentum=0.01, track_running_stats=True)

        self.scene_extractor = Scene(channels=self.scene_channels)
        self.depth_extractor = Depth(channels=self.depth_channels)
        self.normals_extractor = Normals(channels=self.normals_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Shape:
            - X:  :math:`(N, C_in, H, W)` where N is the batch size, and (H, W) is the feature map height and width
            - Output: :math:`(N, C_out, H, W)` where :math:`C_out = C_in`
            - Scene pred: :math:`(N, num_scenes)`
            - Depth pred: :math:`(N, 1, H, W)`
            - Normals pred: :math:`(N, 3, H, W)`
        """

        identity = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        out = identity
        aux_out = []

        if self.scene:
            scene, scene_pred = self.scene_extractor(x)
            aux_out.append(scene_pred)
            out = out + scene
        else:
            aux_out.append(None)

        if self.depth:
            depth, depth_pred = self.depth_extractor(x)
            aux_out.append(depth_pred)
            out = out + depth
        else:
            aux_out.append(None)

        if self.normals:
            normals, normals_pred = self.normals_extractor(x)
            aux_out.append(normals_pred)
            out = out + normals
        else:
            aux_out.append(None)

        out = F.relu(self.bn3(out))

        return (out, *aux_out)
