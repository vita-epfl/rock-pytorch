import math
import random
from typing import Dict, Any, Optional, Tuple

import PIL
import torch
from PIL import Image
from torchvision import transforms as transforms

import rock.ssd.prior_boxes
from rock.ssd.encoder import Encoder
from rock.ssd.encoder import iou


class SSDCropping(object):
    """ Cropping for SSD, according to the original paper, but with fixed aspect ratios

    Randomly choose between the following 3 options:
    1. Keep the original image
    2. Random crop where minimum IoU is randomly chosen between 0.1, 0.3, 0.5, 0.7, 0.9
    3. Random crop
    Modified from https://github.com/chauhan-utk/ssd.DomainAdaptation/blob/master/utils/augmentations.py
    """

    def __init__(self, forced_crops: bool = False) -> None:
        """
        Args:
            forced_crops: force all images to be cropped (default: False)
        """

        self.sample_options = [
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        ]

        if not forced_crops:
            # Add "do nothing"
            self.sample_options.append(None)
            # Make random cropping twice as likely as no cropping by adding a second (None, None) option
            # Increases classification task regularization on the ROCK dataset
            self.sample_options.append((None, None))

    def __call__(self,
                 sample: Dict[str, Any],
                 auxiliary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Crops the given sample

        Args:
            sample: input sample
            auxiliary:  Dict specifying info on input sample to be used for cropping the depth and normals map,
            None if no auxiliary features (default is None)

        Returns:
            random crop of the input sample with fixed aspect ratio
        """

        img = sample['img']
        img_size = sample['size']
        bboxes = sample['bboxes']
        labels = sample['labels']

        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)

            # If no crop or box has no bboxes, don't crop
            if (mode is None) or (bboxes.shape[0] == 0):
                return sample

            htot, wtot = img_size

            min_iou, max_iou = mode
            min_iou = float("-inf") if min_iou is None else min_iou
            max_iou = float("+inf") if max_iou is None else max_iou

            # Implementation uses 30 iterations to find a possible candidate
            for _ in range(30):

                # area of each sampled crop uniformly distributed in[0.1, 1],
                w = math.sqrt(random.uniform(0.1, 1.0))
                h = w

                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                ious = iou(bboxes, torch.tensor([[left, top, right, bottom]]))

                # tailor all the bboxes and return
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5 * (bboxes[:, 1] + bboxes[:, 3])

                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes, continue searching again
                if not masks.any():
                    continue

                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom

                bboxes = bboxes[masks, :]
                labels = labels[masks]

                left_idx = int(left * wtot)
                top_idx = int(top * htot)
                right_idx = int(right * wtot)
                bottom_idx = int(bottom * htot)
                img = img.crop((left_idx, top_idx, right_idx, bottom_idx))

                bboxes[:, 0] = (bboxes[:, 0] - left) / w
                bboxes[:, 1] = (bboxes[:, 1] - top) / h
                bboxes[:, 2] = (bboxes[:, 2] - left) / w
                bboxes[:, 3] = (bboxes[:, 3] - top) / h

                htot = bottom_idx - top_idx
                wtot = right_idx - left_idx

                if auxiliary is not None:
                    ltrb = (left_idx, top_idx, right_idx, bottom_idx)
                    auxiliary['scaling_factor'] = 1 / w
                    sample = self._crop_multi_task(sample, ltrb)

                sample['img'] = img
                sample['size'] = (htot, wtot)
                sample['bboxes'] = bboxes
                sample['labels'] = labels

                return sample

    @staticmethod
    def _crop_multi_task(sample, ltrb):
        if sample['depth']:
            sample['depth'] = sample['depth'].crop(ltrb)

        if sample['normals'] and sample['normals_mask']:
            sample['normals'] = sample['normals'].crop(ltrb)
            sample['normals_mask'] = sample['normals_mask'].crop(ltrb)

        return sample


class RandomHorizontalFlip(object):
    """Horizontally flips a sample with a given probability
    """

    def __init__(self, p: float = 0.5) -> None:
        """
        Args:
            p: flip probability
        """

        self.p = p

    def __call__(self,
                 sample: Dict[str, Any],
                 auxiliary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Args:
            sample (Dict[str, Any]): input sample
            auxiliary (Optional[Dict[str, Any]]): None if no auxiliary features, Dict specifying info on input
                sample otherwise, to be used for flipping the depth and normals map

        Returns:
            (Dict[str, Any]): horizontal flip of the input sample

        """

        img = sample['img']
        bboxes = sample['bboxes']

        if random.random() < self.p:

            # Make sure to only flip boxes if they exist
            if bboxes.shape[0] != 0:
                bboxes[:, 0], bboxes[:, 2] = 1.0 - bboxes[:, 2], 1.0 - bboxes[:, 0]

            img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if auxiliary is not None:
                auxiliary['flip'] = True
                sample = self._flip_multi_task(sample)

        sample['img'] = img
        sample['bboxes'] = bboxes
        return sample

    @staticmethod
    def _flip_multi_task(sample):
        sample['depth'] = sample['depth'].transpose(Image.FLIP_LEFT_RIGHT)
        sample['normals'] = sample['normals'].transpose(Image.FLIP_LEFT_RIGHT)
        sample['normals_mask'] = sample['normals_mask'].transpose(Image.FLIP_LEFT_RIGHT)

        return sample


class DepthTrans(object):
    """Transforms the depth map from a PIL Image to a Tensor matching the size of the backbone output feature map
    """

    def __init__(self,
                 size: Tuple[int, int] = (480, 640),
                 scale: float = 16) -> None:
        """
        Args:
            size: size of the backbone input image
            scale: scaling of the backbone
        """
        self.size = (round(size[0] / scale), round(size[1] / scale))

    def __call__(self,
                 depth: PIL.Image,
                 auxiliary: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            depth: depth image
            auxiliary: dictionary containing info on the auxiliary features

        Returns:
            resized tensor

        """
        trans = transforms.Compose([
            transforms.Resize(self.size, Image.NEAREST),
            transforms.ToTensor()])

        depth = trans(depth)
        depth = depth / auxiliary['scaling_factor']

        return depth


class NormalsTrans(object):
    """Transforms the normals and normals mask

    Transforms the normals and normals mask from PIL Images to normalized Tensors matching the size of
    the backbone output feature map
    """

    def __init__(self,
                 size: Tuple[int, int] = (480, 640),
                 scale: float = 16) -> None:
        """
        Args:
            size (Tuple[int, int]): size of the backbone input image
            scale (int): scaling of the backbone
        """
        self.size = (round(size[0] / scale), round(size[1] / scale))

    @staticmethod
    def normalize(x):
        """Normalizes the normals

        Args:
            x (tensor): input tensor

        Returns:
            (tensor): normalized tensor

        """
        l2_norm = torch.clamp(torch.norm(x, dim=0), min=1e-5)
        x = x / l2_norm
        return x

    def __call__(self,
                 normals: PIL.Image,
                 normals_mask: PIL.Image,
                 auxiliary: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            normals: normals image
            normals_mask: 1-channel normals mask image
            auxiliary: dictionary containing info on the auxiliary features

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing the transformed normals and normals_mask tensors

        """

        trans = transforms.Compose([
            transforms.Resize(self.size, Image.NEAREST),
            transforms.ToTensor()])

        mean = 0.5
        normals = trans(normals) - mean
        normals_mask = trans(normals_mask).type(torch.bool)

        if auxiliary['flip']:
            normals[0, :, :] = -normals[0, :, :]

        normals = self.normalize(normals)

        if auxiliary['scaling_factor'] != 1:
            normals[2, :, :] = normals[2, :, :] * auxiliary['scaling_factor']
            normals = self.normalize(normals)

        return normals, normals_mask


class Transformer(object):
    """ Transform input sample into dict of tensors, and optionally perform data augmentation
    """

    def __init__(self,
                 pboxes: rock.ssd.prior_boxes.PriorBoxes,
                 size: Tuple[int, int] = (480, 640),
                 train: bool = True,
                 forced_crops: bool = False) -> None:
        """
        Args:
            pboxes: prior boxes
            size: input image size (default: (480, 640))
            train: indicate whether to apply train transformations or not (default: True)
            forced_crops: force all train images to be cropped (default: False)
        """

        self.size = size
        self.train = train

        self.pboxes = pboxes
        self.encoder = Encoder(self.pboxes)

        self.crop = SSDCropping(forced_crops)
        self.hflip = RandomHorizontalFlip()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.img_train_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ColorJitter(brightness=0.15, contrast=0.25,
                                   saturation=0.25, hue=0.05),
            transforms.ToTensor(),
            self.normalize
        ])

        self.depth_trans = DepthTrans(size=self.size, scale=self.pboxes.scale_change())
        self.normals_trans = NormalsTrans(size=self.size, scale=self.pboxes.scale_change())

        self.img_test_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            self.normalize])

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs data augmentation on the image

        Keys of dict are: `img`, `img_id`, `size`, `bboxes`, `labels`, `bboxes_mask`, `scene_id`,
        `depth`, `normals`, `normals_mask`, `auxiliary`

        Input types are:
            - `img` (PIL.Image)
            - `img_id` (int)
            - `size` (Tuple[int, int])
            - `bboxes` (torch.Tensor) Shape: :math:`(num_targets, 4)`
            - `labels` (torch.Tensor) Shape: :math:`(num_targets)`
            - `bboxes_mask` (None)
            - `scene_id` (int)
            - `depth` (PIL.Image)
            - `normals` (PIL.Image)
            - `normals_mask` (PIL.Image)
            - `auxiliary` (bool)
        |
        Output types are:
            - `img` (torch.Tensor) Shape: :math:`(3, H_img, W_img)`
            - `img_id` (int)
            - `size` (Tuple[int, int])
            - `bboxes` (torch.Tensor) Shape: :math:`(4, num_priors)`
            - `labels` (torch.Tensor) Shape: :math:`(num_priors)`
            - `bboxes_mask` (torch.BoolTensor) Shape: :math:`(num_priors)`
            - `scene_id` (int)
            - `depth` (torch.Tensor) Shape: :math:`(1, H_featuremap, W_featuremap)`
            - `normals` (torch.Tensor) Shape: :math:`(3, H_featuremap, W_featuremap)`
            - `normals_mask` (torch.BoolTensor) Shape: :math:`(1, H_featuremap, W_featuremap)`
            - `auxiliary` (bool)

        Args:
            sample: dictionary containing the image, bounding boxes, labels, bounding boxes masks,
                depth map, normals and normals mask of a sample from the NYUv2 dataset

        Returns:
            input sample, with data augmentation and transformation applied

        """
        s = sample

        if s['auxiliary']:
            auxiliary = {'flip': False, 'scaling_factor': 1}
        else:
            auxiliary = None

        # Train transform
        if self.train:
            # Crop and flip
            s = self.crop(s, auxiliary=auxiliary)
            s = self.hflip(s, auxiliary=auxiliary)
            # Apply training image transform
            s['img'] = self.img_train_trans(s['img']).contiguous()

        # Test transform
        else:
            # Apply test image transform
            s['img'] = self.img_test_trans(s['img']).contiguous()

        # Encode the bounding boxes
        s['bboxes'], s['labels'], s['bboxes_mask'] = self.encoder.encode(s['bboxes'], s['labels'])
        # permute to get bboxes in the correct format (4, num_priors) instead of (num_priors, 4)
        s['bboxes'] = s['bboxes'].permute(1, 0)

        # Transform the auxiliary tasks
        if s['auxiliary']:
            if s['depth']:
                s['depth'] = self.depth_trans(s['depth'], auxiliary)

            if s['normals'] and s['normals_mask']:
                s['normals'], s['normals_mask'] = self.normals_trans(s['normals'], s['normals_mask'], auxiliary)

        return s
