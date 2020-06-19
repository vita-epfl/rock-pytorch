import torch
import torchvision
from torch import nn
from torchvision.models import ResNet
# noinspection PyProtectedMember
from torchvision.models.resnet import Bottleneck


class ResNet50(ResNet):
    """Custom ResNet50 used as the network backbone

    Modified version of the base ResNet50 from torchvision,
    found at: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    This ResNet replace stride with a dilation on layer 4 for a larger feature map, and
    removes the average pooling and fully-connected part of the network.
    During training, BatchNorm is frozen.
    """

    def __init__(self) -> None:
        # Replace stride with dilation on layer 4
        super().__init__(Bottleneck, [3, 4, 6, 3], replace_stride_with_dilation=[False, False, True])
        self.training = True

        # Pretrain the network
        self.load_state_dict(torchvision.models.resnet50(pretrained=True).state_dict())

        # Replace last 2 (unused) layers with identity
        self.avgpool = nn.Identity()
        self.fc = nn.Identity()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward implementation of ResNet50 where avg_pool, flatten and fc are removed

        Shape:
            - X: :math:`(N, 3, H_in, W_in)` where :math:`(H_in, W_in)` are the image height and width
            - Output: :math:`(N, C_out, H_out, W_out)` where :math:`C_out` is the number of output channels
        |
        For ROCK implementation on the NYUv2 dataset:
            - :math:`(H_in, W_in)  = (480, 640)`
            - :math:`(C_out, H_out, W_out) = (2048, 30, 40)`
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def freeze_bn(self) -> None:
        """Freezes batch norm layers
        """

        for module in self.modules():
            if type(module) == nn.modules.batchnorm.BatchNorm2d:
                module.eval()

    def train(self, mode=True):
        r"""Override of nn.module method to freeze batchnorm in all cases

        Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)

        self.freeze_bn()

        return self

