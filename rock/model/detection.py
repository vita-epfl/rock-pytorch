from typing import List, Tuple, Optional

import torch
import torch.nn as nn


class Detection(nn.Module):
    """SSD detection layers

    Consists of 6 detection layers (by default), used for object detection and classification.
    """

    def __init__(self,
                 additional_layers: Optional[torch.nn.ModuleList] = None) -> None:
        """ Initializes the detection layers, uses the default additional layers if ``additional_layers`` is None
        """
        super().__init__()

        self.out_channels = [2048, 1024, 512, 256, 256, 256]
        self.num_priors = [4, 6, 6, 6, 4, 4]

        self.num_labels = 20  # number of ROCK classes

        if additional_layers:
            self.additional_layers = additional_layers
        else:
            self.additional_layers = self._build_default_additional_layers()

        self.loc, self.conf = self._build_loc_and_conf()

        self._init_weights()
        self._init_bias()

    def _build_default_additional_layers(self) -> nn.ModuleList:
        """ Constructs the default additional layers for the ROCK implementation on the NYUv2 dataset
        """
        additional_layers = []
        mid_layer_channels = [1024, 256, 128, 128, 128]

        for i, (input_size, output_size, channels) in enumerate(
                zip(self.out_channels[:-1], self.out_channels[1:], mid_layer_channels)):
            if i <= 2:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1),
                    nn.BatchNorm2d(channels, momentum=0.01, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(output_size, momentum=0.01, track_running_stats=True),
                    nn.ReLU(inplace=True),
                )
            elif i == 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1),
                    nn.BatchNorm2d(channels, momentum=0.01, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3),
                    nn.BatchNorm2d(output_size, momentum=0.01, track_running_stats=True),
                    nn.ReLU(inplace=True),
                )
            else:
                # This layer goes from (N, C_in, 2,3) to (N, C_out, 1,1).
                # If input is of another shape (such as square), change the
                # kernel size of this layer in order to obtain a (1, 1) output
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1),
                    nn.BatchNorm2d(channels, momentum=0.01, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=(2, 3)),
                    nn.BatchNorm2d(output_size, momentum=0.01, track_running_stats=True),
                    nn.ReLU(inplace=True),
                )

            additional_layers.append(layer)

        return nn.ModuleList(additional_layers)

    def _build_loc_and_conf(self):
        loc = []
        conf = []
        for num_prior, out_channel in zip(self.num_priors, self.out_channels):
            loc.append(nn.Conv2d(out_channel, num_prior * 4, kernel_size=3, padding=1))
            conf.append(nn.Conv2d(out_channel, num_prior * self.num_labels, kernel_size=3, padding=1))

        loc = nn.ModuleList(loc)
        conf = nn.ModuleList(conf)

        return loc, conf

    def _init_weights(self) -> None:
        """ Weight initialization for additional layers
        """
        layers = [*self.additional_layers, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                # Switch from xavier_uniform_ to kaiming_normal_
                if param.dim() > 1:
                    nn.init.kaiming_normal_(param, nonlinearity="relu")

    def _init_bias(self) -> None:
        """ Bias initialization for confidence layers

        As indicated in the paper "Focal Loss for Dense Object Detection" https://arxiv.org/abs/1708.02002
        """

        final_conf_layers = self._get_final_conf_layers()
        for layer in final_conf_layers:
            pi = 0.01
            fg_value = -torch.log(torch.tensor((1 - pi) / pi)).item()
            bias = layer.bias.data
            bias = torch.zeros_like(bias)
            bias = bias.reshape((self.num_labels, -1))
            bias[1:, :] = fg_value
            layer.bias.data = bias.reshape(-1)

    def _get_final_conf_layers(self) -> List[nn.Module]:
        """ Gets the final conf layer (the one that outputs the result) for each detection layer
        """
        conf_layers = [*self.conf]
        final_conf_layers = []

        for layer in conf_layers:
            # Conf layer can either be one convolutional layer or a sequence of layers
            if isinstance(layer, torch.nn.Sequential):
                final_conf_layers.append(layer[-1])
            else:
                final_conf_layers.append(layer)

        return final_conf_layers

    def _reshape_bbox(self,
                      detection_feed: List[torch.Tensor],
                      loc: torch.nn.modules.container.ModuleList,
                      conf: torch.nn.modules.container.ModuleList) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshapes the input to correspond to the prior boxes
        """
        locs = []
        confs = []
        for x, l, c in zip(detection_feed, loc, conf):
            loc_out = l(x).reshape((x.shape[0], 4, -1))
            locs.append(loc_out)

            conf_out = c(x).reshape((x.shape[0], self.num_labels, -1))
            confs.append(conf_out)

        locs = torch.cat(locs, dim=-1).contiguous()
        confs = torch.cat(confs, dim=-1).contiguous()
        return locs, confs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward implementation of the SSD detection layers

        Shape:
            - X: :math:`(N, C_in, H_in, W_in)` where N is the batch size
            - Locs: :math:`(N, 4, num_priors)` where :math:`num_priors` is the number of priors
            - Confs: :math:`(N, num_labels, num_priors)` where :math:`num_labels` is the number of object labels
        |
        For ROCK implementation on the NYUv2 dataset:
            - :math:`C_in = 2048`
            - :math:`(H_in, W_in)  = (30, 40)`
            - :math:`num_priors = 7228`
            - :math:`num_labels = 20`
        """
        detection_feed = [x]
        for layer in self.additional_layers:
            x = layer(x)
            detection_feed.append(x)

        locs, confs = self._reshape_bbox(detection_feed, self.loc, self.conf)

        return locs, confs

