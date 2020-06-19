import itertools
from math import sqrt
from typing import Tuple, List

import numpy as np
import torch


class PriorBoxes(object):
    """ Prior boxes of the network

    Modified from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD for
    more modularity and support for rectangular images
    """

    def __init__(self, fig_size: Tuple[int, int],
                 feat_size: List[Tuple[int, int]],
                 steps: Tuple[List[int], List[int]],
                 sk: List[float],
                 aspect_ratios: List[List[int]],
                 variance_xy: float = 0.1,
                 variance_wh: float = 0.2) -> None:
        """

        Args:
            fig_size: size of input image
            feat_size: list of sizes of the feature maps
            steps: scale change for each feature map
            sk: scale of the prior boxes for each feature map
            aspect_ratios: aspect ratio of the prior boxes for each feature map
            variance_xy: default value 0.1
            variance_wh: default value 0.2
        """

        self.feat_size = feat_size
        self.fig_height, self.fig_width = fig_size

        # More info: https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/
        self.variance_xy_ = variance_xy
        self.variance_wh_ = variance_wh

        self.steps_height, self.steps_width = steps
        self.sk = sk

        fk_height = self.fig_height / np.array(self.steps_height)
        fk_width = self.fig_width / np.array(self.steps_width)
        self.aspect_ratios = aspect_ratios

        self.prior_boxes = []

        # Incorporate different features sizes for width and height
        for idx, (sfeat_height, sfeat_width) in enumerate(self.feat_size):

            sk1 = self.sk[idx]
            sk2 = self.sk[idx + 1]
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:

                # Iterate over product of width and height
                for i, j in itertools.product(range(sfeat_height), range(sfeat_width)):
                    cx, cy = (j + 0.5) / fk_width[idx], (i + 0.5) / fk_height[idx]
                    self.prior_boxes.append((cx, cy, w, h))

        self.pboxes = torch.tensor(self.prior_boxes, dtype=torch.float)
        self.pboxes.clamp_(min=0, max=1)

        # For IoU calculation
        self.pboxes_ltrb = self.pboxes.clone()
        self.pboxes_ltrb[:, 0] = self.pboxes[:, 0] - 0.5 * self.pboxes[:, 2]
        self.pboxes_ltrb[:, 1] = self.pboxes[:, 1] - 0.5 * self.pboxes[:, 3]
        self.pboxes_ltrb[:, 2] = self.pboxes[:, 0] + 0.5 * self.pboxes[:, 2]
        self.pboxes_ltrb[:, 3] = self.pboxes[:, 1] + 0.5 * self.pboxes[:, 3]

    @property
    def variance_xy(self) -> float:
        """  More info: https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/
        """
        return self.variance_xy_

    @property
    def variance_wh(self) -> float:
        """  More info: https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/
        """
        return self.variance_wh_

    def scale_change(self) -> float:
        """ Scale between input image and feature map
        """
        return sqrt(self.steps_height[0] * self.steps_width[0])

    def __call__(self, order: str = "ltrb") -> torch.Tensor:
        if order == "ltrb":
            return self.pboxes_ltrb
        if order == "xywh":
            return self.pboxes


def pboxes_rock() -> PriorBoxes:
    """Prior boxes for the NYUv2 dataset

    Returns:
        prior boxes for the given specifications
    """
    figsize = (480, 640)
    feat_size = [(30, 40), (15, 20), (8, 10), (4, 5), (2, 3), (1, 1)]

    # steps is [figsize_h/(feat_size[i][h]), figsize_w/(feat_size[i][w]))]
    steps_h = [16, 32, 60, 120, 240, 480]
    steps_w = [16, 32, 64, 128, 213, 640]
    steps = (steps_h, steps_w)

    # 6 layers: conv4_3 = 0.07, smin=0.15, smax=1.05
    sk = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]

    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    pboxes = PriorBoxes(figsize, feat_size, steps, sk, aspect_ratios)
    return pboxes
