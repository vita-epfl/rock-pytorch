from typing import Tuple, List, Union

import torch
import torch.nn.functional as F

import rock.ssd.prior_boxes

DecoderOutput = Union[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], None]


def iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

    Shape:
    - box1: :math:`(N, 4)`
    - box2: :math:`(M, 4)`
    - Output: :math:`(N, M)`

    Modified from: https://github.com/kuangliu/pytorch-ssd
   """
    N = box1.shape[0]
    M = box2.shape[0]

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou_out = inter / (area1 + area2 - inter)
    return iou_out


class Encoder:
    """ Encodes and decodes from (coordinates, labels) bounding boxes to SSD prior boxes

    Code modified from:
        - https://github.com/lufficc/SSD
        - https://github.com/kuangliu/pytorch-ssd
        - https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
        - https://github.com/amdegroot/ssd.pytorch
    """

    def __init__(self, pboxes: rock.ssd.prior_boxes.PriorBoxes) -> None:
        self.pboxes = pboxes(order="ltrb")
        self.pboxes_xywh = pboxes(order="xywh").unsqueeze(dim=0)
        self.num_priors = self.pboxes.shape[0]
        self.variance_xy = pboxes.variance_xy
        self.variance_wh = pboxes.variance_wh

    def area_of(self,
                left_top: torch.Tensor,
                right_bottom: torch.Tensor) -> torch.Tensor:
        """Compute the areas of rectangles given two corners.

        Shape:
            - left_top: :math:`(*, 2)` where * means any number of dimensions
            - right_bottom: :math:`(*, 2)` where * from left_top = * from right_bottom
            - area (out): :math:`(*)` where * from out = * from left_top

        :math:`*` is a constant here and is the same for left_top, right_bottom and out

         Modified from: https://github.com/lufficc/SSD
        """
        hw = torch.clamp(right_bottom - left_top, min=0.0)
        return hw[..., 0] * hw[..., 1]

    def iou_of(self,
               boxes0: torch.Tensor,
               boxes1: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Return intersection-over-union (Jaccard index) of boxes.

        Shape:
            - boxes0: :math:`(N1, M1, 4)`
            - boxes1 :math: (N2, M2, 4)
            - out: :math:`(N, M)` where N = max(N1, N2) and M = max(M1, M2)

        Modified from: https://github.com/lufficc/SSD
        """
        overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def encode(self,
               gt_boxes: torch.Tensor,
               gt_labels: torch.Tensor,
               iou_bg_threshold: float = 0.4,
               iou_box_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign ground truth boxes and targets to priors.

        Shape:
            - gt_boxes: :math:`(num_targets, 4)`
            - gt_labels: :math:`(num_targets)`
            - boxes (out): :math:`(num_priors, 4)`
            - labels (out): :math:`(num_priors)`
            - mask (out): :math:`(num_priors)`

        Modified from assign_priors function of: https://github.com/lufficc/SSD
        """
        # If the image has no boxes, return the default
        if gt_boxes.shape[0] == 0:
            return self._no_box_encode()

        corner_form_priors = self.pboxes

        ious = self.iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
        # size: num_priors
        best_target_per_prior, best_target_per_prior_index = ious.max(1)
        # size: num_targets
        best_prior_per_target, best_prior_per_target_index = ious.max(0)

        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index

        # size: num_priors
        labels = gt_labels[best_target_per_prior_index]

        # 2.0 is used to make sure every target has a prior assigned
        best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
        labels[best_target_per_prior < iou_bg_threshold] = 0  # the background id
        boxes = gt_boxes[best_target_per_prior_index]

        # Add a mask to exclude certain priors within a threshold
        bg_mask = best_target_per_prior < iou_bg_threshold
        box_mask = best_target_per_prior > iou_box_threshold
        mask = bg_mask | box_mask

        x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        y = 0.5 * (boxes[:, 1] + boxes[:, 3])
        w = -boxes[:, 0] + boxes[:, 2]
        h = -boxes[:, 1] + boxes[:, 3]

        boxes[:, 0] = x
        boxes[:, 1] = y
        boxes[:, 2] = w
        boxes[:, 3] = h

        return boxes, labels, mask

    def _no_box_encode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Assign correct priors if there are no bounding boxes in the image
        """
        labels = torch.zeros(self.num_priors, dtype=torch.long)
        mask = torch.ones(self.num_priors, dtype=torch.bool)
        boxes = self.pboxes.clone()
        # Transform format to xywh format
        x, y, w, h = 0.5 * (boxes[:, 0] + boxes[:, 2]), \
                     0.5 * (boxes[:, 1] + boxes[:, 3]), \
                     -boxes[:, 0] + boxes[:, 2], \
                     -boxes[:, 1] + boxes[:, 3]
        boxes[:, 0] = x
        boxes[:, 1] = y
        boxes[:, 2] = w
        boxes[:, 3] = h

        return boxes, labels, mask

    def transform_back_batch(self,
                             boxes: torch.Tensor,
                             scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Bounding box format transformation from network output to format for non-max suppression

        Applies back variance_xy and variance_wh, permutes dims of boxes, switches from xywh to ltrb
        and applies softmax to the scores

         Shape:
            - boxes (input): :math:`(N, 4, num_priors)` where N is the batch_size
            - scores (input): :math:`(N, num_labels, num_priors)`
            - boxes (out): :math:`(N, num_priors, 4)`
            - scores (out): :math:`(N, num_priors, num_labels)`

        Modified from: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
        """

        if boxes.device != self.pboxes.device:
            self.pboxes = self.pboxes.to(boxes.device)
            self.pboxes_xywh = self.pboxes_xywh.to(boxes.device)

        boxes = boxes.permute(0, 2, 1)
        scores = scores.permute(0, 2, 1)

        boxes[:, :, :2] = self.variance_xy * boxes[:, :, :2]
        boxes[:, :, 2:] = self.variance_wh * boxes[:, :, 2:]

        boxes[:, :, :2] = boxes[:, :, :2] * self.pboxes_xywh[:, :, 2:] + self.pboxes_xywh[:, :, :2]
        boxes[:, :, 2:] = boxes[:, :, 2:].exp() * self.pboxes_xywh[:, :, 2:]

        # Transform format to ltrb
        l, t, r, b = boxes[:, :, 0] - 0.5 * boxes[:, :, 2], \
                     boxes[:, :, 1] - 0.5 * boxes[:, :, 3], \
                     boxes[:, :, 0] + 0.5 * boxes[:, :, 2], \
                     boxes[:, :, 1] + 0.5 * boxes[:, :, 3]

        boxes[:, :, 0] = l
        boxes[:, :, 1] = t
        boxes[:, :, 2] = r
        boxes[:, :, 3] = b

        return boxes, F.softmax(scores, dim=-1)

    def decode_batch(self,
                     boxes: torch.Tensor,
                     scores: torch.Tensor,
                     nms_iou_threshold: float = 0.3,
                     max_output_num: int = 200) -> DecoderOutput:
        """ Decode network prediction tensor and obtain bounding boxes location, label and confidence

        Shape:
            - boxes (input): :math:`(N, 4, num_priors)` where N is the batch size
            - scores (input): :math:`(N, num_labels, num_priors)`
            - location (out): :math:`(M, 4)` where M is the number of detected boxes
            - label (out):  :math:`(M)`
            - confidence (out):  :math:`(M)`

        Output is a list of length `batch_size` containing tuples of location, label and confidence tensors

         Modified from: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
        """
        bboxes, probs = self.transform_back_batch(boxes, scores)

        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, nms_iou_threshold, max_output_num))
        return output

    @staticmethod
    def decode_single(boxes: torch.Tensor,
                      scores: torch.Tensor,
                      nms_iou_threshold: float,
                      max_output: int,
                      max_num: int = 200) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs non-maximum suppression and returns the decoded bounding boxes

        Shape:
            - boxes (input): :math:`(num_priors, 4)`
            - scores (input): :math:`(num_priors, num_labels)`
            - location (out): :math:`(M, 4)` where M is the number of detected boxes
            - label (out):  :math:`(M)`
            - confidence (out):  :math:`(M)`

        Modified from: https://github.com/amdegroot/ssd.pytorch
        """
        boxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores.split(1, 1)):
            # skip background
            if i == 0:
                continue

            score = score.squeeze(1)
            mask = score > 0.05

            bboxes, score = boxes[mask, :], score[mask]
            if score.shape[0] == 0:
                continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = iou(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < criteria
                score_idx_sorted = score_idx_sorted[iou_sorted < nms_iou_threshold]
                candidates.append(idx)

            boxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        if not boxes_out:
            out = (torch.tensor([]), torch.tensor([]), torch.tensor([]))
            return out

        boxes_out = torch.cat(boxes_out, dim=0)
        labels_out = torch.tensor(labels_out, dtype=torch.long)
        scores_out = torch.cat(scores_out, dim=0)

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]

        return boxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]
