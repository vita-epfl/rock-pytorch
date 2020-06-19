from typing import Optional, Tuple

import torch
import torch.nn as nn

import rock.model.auxiliary
import rock.model.backbone
import rock.model.detection


class Network(nn.Module):
    """Network class which combines the backbone, the auxiliary tasks (optional), and the detection block

    Can be used to implement the ROCK network architecture or a baseline Single Shot Detector
    """

    def __init__(self,
                 backbone: torch.nn.Module,
                 detection: torch.nn.Module,
                 auxiliary: Optional[torch.nn.Module] = None) -> None:
        """
        Args:
            backbone: backbone used to obtain the base feature map
            detection: detection layer of Network
            auxiliary: auxiliary block used for MTL
        """
        super().__init__()

        self.feature_extractor = backbone
        self.detection = detection
        self.auxiliary = auxiliary

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """ Forward implementation of model

        Gives a tuple containing the location and confidence tensors, as well as the scene, depth and normals tensors
        if auxiliary tasks are added

        Shape:
            - X: :math:`(N, 3, H_in, W_in)` where :math:`(H_in, W_in)` are the image height and width
            - Locs: :math:`(N, 4, num_priors)` where :math:`num_priors` is the number of priors
            - Confs: :math:`(N, num_labels, num_priors)` where :math:`num_labels` is the number of object labels
            - Scene pred: :math:`(N, num_scenes)`
            - Depth pred: :math:`(N, 1, H_featuremap, W_featuremap)`
            - Normals pred: :math:`(N, 3, H_featuremap, W_featuremap)`
        |
        For ROCK implementation on the NYUv2 dataset:
            - :math:`(H_in, W_in)  = (480, 640)`
            - :math:`num_priors = 7228`
            - :math:`num_labels = 20`
            - :math:`num_scenes = 27`
            - :math:`(H_featuremap, W_featuremap) = (30, 40)`
        """
        x = self.feature_extractor(x)

        aux_out = []
        if self.auxiliary:
            # For ROCK without fusion, replace x by _
            x, scene, depth, normals = self.auxiliary(x)
            aux_out.extend([scene, depth, normals])

        locs, confs = self.detection(x)

        return (locs, confs, *aux_out)


def rock_network(aux_tasks: Tuple[str] = ('scene', 'depth', 'normals')) -> torch.nn.Module:
    """
    Creates a model similar to the one described in the paper
    "Revisiting Multi-Task Learning with ROCK: a Deep Residual Auxiliary Block for Visual Detection"
    """
    backbone = rock.model.backbone.ResNet50()
    detection = rock.model.detection.Detection()
    auxiliary = rock.model.auxiliary.ROCK(aux_tasks)
    model = Network(backbone=backbone, detection=detection, auxiliary=auxiliary)
    return model


def baseline_ssd() -> torch.nn.Module:
    """
    Creates a network similar to the one described in the paper
    "Revisiting Multi-Task Learning with ROCK: a DeepResidual Auxiliary Block for Visual Detection",
    but with no auxiliary block (ROCK). This network is based on the SSD (Single Shot Detector) architecture
    """
    backbone = rock.model.backbone.ResNet50()
    detection = rock.model.detection.Detection()
    model = Network(backbone=backbone, detection=detection, auxiliary=None)
    return model
