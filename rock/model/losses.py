from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import rock.ssd.prior_boxes


def normalize(x: torch.Tensor) -> torch.Tensor:
    """ Normalizes tensor for surface normals

   Shape:
        - X: :math:`(N, *)` where N is the batch size and * means, any number of additional dimensions ≥1
        - Output: :math:`(N, *)`, same size as the input

    """
    l2_norm = torch.clamp(torch.norm(x, dim=1, keepdim=True), min=1e-5)
    x = x / l2_norm
    return x


def reverse_huber_loss(x: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    """ Computes the reverse huber loss in log space

    Shape:
        - X: :math:`(N, C, H, W)` where N is the batch size and C is the number of channels (1 for depth)
        - Target: :math:`(N, C, H, W)`
        - Output: scalar
    """
    # Reverse huber loss in log space
    x = torch.abs(x - torch.log(torch.clamp(target, min=1e-2, max=10)))

    # Fix c to a specific value (can be changed)
    c = 0.5
    gt_c = (x > c).float()
    le_c = (x <= c).float()

    linear = x * le_c
    quadratic = (((x ** 2) + (c ** 2)) / (2 * c)) * gt_c
    loss = linear + quadratic
    loss = torch.mean(loss)
    return loss


def huber_loss(x: torch.Tensor,
               target: torch.Tensor) -> torch.Tensor:
    """ Computes a Smooth-L1 loss (Huber loss) in log space

    Shape:
        - X: :math:`(N, C, H, W)` where N is the batch size and C is the number of channels (1 for depth)
        - Target: :math:`(N, C, H, W)`
        - Output: scalar
    """
    log_target = torch.log(torch.clamp(target, min=1e-3, max=10))
    smooth_l1 = nn.SmoothL1Loss(reduction='mean')
    loss = smooth_l1(x, log_target)
    return loss


def surface_normals_loss(x: torch.Tensor,
                         target: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """ Computes the surface normals loss by combining the dot_product with the L2 loss

    Shape:
        - X: :math:`(N, C, H, W)` where N is the batch size and C is the number of channels (3 for normals)
        - Target: :math:`(N, C, H, W)`
        - Mask: :math:`(N, 1, H, W)`
        - Output: scalar
    """

    x = normalize(x)

    # Set the minimum dot product value to 0
    dot_product = -torch.sum(x * target, dim=1) + 1
    l2_loss = torch.sum((x - target) ** 2, dim=1)

    loss = (dot_product + l2_loss) * mask.squeeze(dim=1).float()
    mean = torch.sum(loss) / torch.sum(mask.float())
    return mean


def scene_cross_entropy_loss(scene_pred: torch.Tensor,
                             scene_gt: torch.Tensor) -> torch.Tensor:
    """ Computes the cross entropy loss for scenes where scene_pred is a log softmax input using NLL loss

    Shape:
        - scene_pred: :math:`(N, C)` where N is the batch size and C is the number of scene types in the dataset
        - scene_gt: :math:`(N)`
        - Output: scalar

    """
    cross_entropy_loss = nn.NLLLoss(reduction='mean')
    loss = cross_entropy_loss(scene_pred, scene_gt)

    return loss


class MultiTaskLoss(nn.Module):
    """ Implements the multi-task loss as the sum of the following:
        1. Scene Loss: (weight 3)
        2. Depth Loss: (weight 3)
        3. Normals Loss: (weight 30)
    """

    def __init__(self,
                 aux_tasks: Tuple[str] = ('scene', 'depth', 'normals')) -> None:
        super(MultiTaskLoss, self).__init__()

        self.scene = 'scene' in aux_tasks
        self.depth = 'depth' in aux_tasks
        self.normals = 'normals' in aux_tasks

        self.scene_weight = 3
        self.depth_weight = 3
        self.normals_weight = 3 * 10

    def forward(self,
                sample: Dict[str, torch.Tensor],
                loss_dict: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """ Forward method of Multi-task loss

        Keys of dict are: `scene_pred`, `scene_gt`, `depth_pred`,
        `depth_gt`, `normals_pred`, `normals_gt`, `normals_mask`

       Shape:
           - scene_pred: :math:`(N, num_scenes)` where N is the batch size & :math:`num_scenes` is the number of scenes
           - scene_gt: :math:`(N)`
           - depth_pred: :math:`(N, 1, H, W)` where :math:`H, W` is the height and width respectively
           - depth_gt: :math:`(N, 1, H, W)`
           - normals_pred: :math:`(N, 3, H, W)`
           - normals_gt: :math:`(N, 3, H, W)`
           - normals_mask: :math:`(N, 1, H, W)`
           - Output: scalar
        """
        losses = []

        if self.scene:
            scene_pred, scene_gt = sample['scene_pred'], sample['scene_gt']
            scene_loss = scene_cross_entropy_loss(scene_pred, scene_gt)
            losses.append(self.scene_weight * scene_loss)

            if loss_dict is not None:
                loss_dict['z_scene_loss'] += scene_loss.item()

        if self.depth:
            depth_pred, depth_gt = sample['depth_pred'], sample['depth_gt']
            depth_loss = huber_loss(depth_pred, depth_gt)
            losses.append(self.depth_weight * depth_loss)

            if loss_dict is not None:
                loss_dict['z_depth_loss'] += depth_loss.item()

        if self.normals:
            normals_pred, normals_gt, normals_mask = sample['normals_pred'], sample['normals_gt'], sample[
                'normals_mask']
            normals_loss = surface_normals_loss(normals_pred, normals_gt, normals_mask)
            losses.append(self.normals_weight * normals_loss)

            if loss_dict is not None:
                loss_dict['z_normals_loss'] += normals_loss.item()

        total_loss = torch.sum(torch.stack(losses))

        return total_loss


class DetectionLoss(nn.Module):
    """ Implements the detection loss as the sum of the following:

        1. Confidence Loss: All labels, with hard negative mining if using _hard_neg_mining_conf_loss \
        or with all background labels if using _all_priors_conf_loss (weight 1)

        2. Localization Loss: Only on positive labels (weight 6)

        Suppose input pboxes has the shape 7228x4

        Modified from: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
    """

    def __init__(self, pboxes: rock.ssd.prior_boxes.PriorBoxes, use_all_priors_conf_loss: bool = False) -> None:
        super(DetectionLoss, self).__init__()

        self.huber_loss = nn.SmoothL1Loss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.use_all_priors_conf_loss = use_all_priors_conf_loss

        self.loc_weight = 6
        self.conf_weight = 1

        self.variance_xy = pboxes.variance_xy
        self.variance_wh = pboxes.variance_wh
        self.pboxes = nn.Parameter(pboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False)

    def _loc_gt(self, loc: torch.Tensor) -> torch.Tensor:
        """Generate Location Vectors

        Shape:
            - loc: :math:`(N, 4, num_priors)` where N is the batch size
            - Output: :math:`(N, 4, num_priors)`
        """

        gxy = loc[:, :2, :] - self.pboxes[:, :2, :]
        gxy = gxy / (self.variance_xy * self.pboxes[:, 2:, :])

        gwh = loc[:, 2:, :] / self.pboxes[:, 2:, :]
        gwh = torch.log(gwh) / self.variance_wh

        return torch.cat((gxy, gwh), dim=1).contiguous()

    def _loc_loss(self,
                  bbox_mask: torch.Tensor,
                  bg_mask: torch.Tensor,
                  gloc: torch.Tensor,
                  ploc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Calculates the bounding box localization loss

        Returns the localization loss and the number of positive prior boxes per sample
        """
        # Loc loss
        loc_gt = self._loc_gt(gloc)
        loc_loss_mask = (bg_mask & bbox_mask).float()
        pos_num = loc_loss_mask.sum(dim=1)
        loc_loss = self.huber_loss(ploc, loc_gt).sum(dim=1)
        loc_loss = (loc_loss_mask * loc_loss).sum(dim=1)
        return loc_loss, pos_num

    def _hard_neg_mining_conf_loss(self, bbox_mask: torch.Tensor,
                                   bg_mask: torch.Tensor,
                                   glabel: torch.Tensor,
                                   plabel: torch.Tensor,
                                   pos_num: torch.Tensor) -> torch.Tensor:
        """ Confidence loss using hard negative mining, used in both the SSD and ROCK paper
        """
        conf = self.cross_entropy_loss(plabel, glabel)
        conf = conf * bbox_mask.float()
        conf_neg = conf.clone()
        conf_neg[bg_mask] = 0
        _, conf_idx = conf_neg.sort(dim=1, descending=True)
        _, conf_rank = conf_idx.sort(dim=1)
        # number of negative three times positive
        neg_num = torch.clamp(3 * pos_num, max=bg_mask.shape[1]).unsqueeze(-1)
        neg_mask = (conf_rank < neg_num)
        conf_loss_mask = (bg_mask | neg_mask).float()
        conf_loss = (conf * conf_loss_mask).sum(dim=1)
        return conf_loss

    def all_priors_conf_loss(self, bbox_mask, bg_mask, glabel, plabel, pos_num):
        """ Confidence loss where all positive and negative priors are used, but weighted differently based on
            the proportion of positive priors compared to negative priors
        """
        conf = self.cross_entropy_loss(plabel, glabel)
        conf = conf * bbox_mask.float()
        neg_num = (~bg_mask).float().sum(dim=1)

        # Multiply negative examples by the proportion of positive examples
        neg_weight = pos_num / neg_num
        # Multiply negative examples loss by 5 to speed up learning a bit, but not too much
        conf_neg = 5 * neg_weight * (conf * (~bg_mask).float()).sum(dim=1)
        conf_pos = (conf * bg_mask.float()).sum(dim=1)

        conf_loss = conf_pos + conf_neg
        return conf_loss

    def forward(self,
                sample: Dict[str, torch.Tensor],
                loss_dict: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """ Forward method of the detection loss

        Keys of dict are: `ploc`, `plabel`, `bboxes`, `labels`, `bboxes_mask`

       Shape:
           - ploc: :math:`(N, 4, num_priors)` where N is the batch size and :math:`num_priors` is the number of priors
           - plabel: :math:`(N, num_labels, num_priors)` where :math:`num_labels` is the number of object labels
           - gloc: :math:`(N, 4, num_priors)`
           - glabel: :math:`(N, num_priors)`
           - bboxes_mask: :math:`(N, num_priors)`
           - Output: scalar
        """
        ploc = sample['ploc']
        plabel = sample['plabel']
        gloc = sample['bboxes']
        glabel = sample['labels']
        bbox_mask = sample['bboxes_mask']

        bg_mask = glabel > 0

        loc_loss, pos_num = self._loc_loss(bbox_mask, bg_mask, gloc, ploc)

        # Classification loss

        if self.use_all_priors_conf_loss:
            # all priors loss
            conf_loss = self.all_priors_conf_loss(bbox_mask, bg_mask, glabel, plabel, pos_num)
        else:
            # hard negative mining loss
            conf_loss = self._hard_neg_mining_conf_loss(bbox_mask, bg_mask, glabel, plabel, pos_num)

        # Sum losses together
        # Weight losses differently
        total_loss = self.loc_weight * loc_loss + self.conf_weight * conf_loss
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        total_loss = (total_loss * num_mask / pos_num).mean(dim=0)

        if loss_dict is not None:
            loss_dict['z_loc_loss'] += (loc_loss * num_mask / pos_num).mean(dim=0).item()
            loss_dict['z_conf_loss'] += (conf_loss * num_mask / pos_num).mean(dim=0).item()

        return total_loss


class Loss(nn.Module):
    """ Combines the detection loss with the multi-task loss, if existing
    """

    def __init__(self, pboxes: rock.ssd.prior_boxes.PriorBoxes,
                 auxiliary: bool = True,
                 aux_tasks: Tuple[str] = ('scene', 'depth', 'normals'),
                 use_all_priors_conf_loss: bool = False) -> None:
        super(Loss, self).__init__()

        self.detection_loss = DetectionLoss(pboxes, use_all_priors_conf_loss=use_all_priors_conf_loss)
        self.multi_task_loss = MultiTaskLoss(aux_tasks=aux_tasks)

        self.auxiliary = auxiliary
        self.aux_tasks = aux_tasks

    def forward(self,
                sample: Dict[str, torch.Tensor],
                loss_dict: Dict[str, float] = None) -> torch.Tensor:
        """ Forward method of the loss

                Keys of dict are: `ploc`, `plabel`, `bboxes`, `labels`, `bboxes_mask`
                and if auxiliary is `True`, keys related to auxiliary tasks are:
                `scene_pred`, `scene_gt`, `depth_pred`, `depth_gt`, `normals_pred`, `normals_gt`, `normals_mask`

        Shape:
            - Input: given in respective loss functions
            - Output: scalar
        """

        if self.auxiliary and self.aux_tasks:
            detection_loss = self.detection_loss(sample, loss_dict)
            multi_task_loss = self.multi_task_loss(sample, loss_dict)
            loss = detection_loss + multi_task_loss
        else:
            loss = self.detection_loss(sample, loss_dict)

        return loss
