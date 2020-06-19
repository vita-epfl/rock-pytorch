import os
import pickle
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils import data as data

import rock.datasets.transforms


class NYUv2Detection(data.Dataset):
    """Datareader for the NYUv2 Dataset
    """

    def __init__(self,
                 path: str,
                 transform: rock.datasets.transforms.Transformer,
                 auxiliary: bool = True) -> None:
        """
        Args:
            path: path to the folder containing the pre-processed dataset
            transform: input transformer
            auxiliary: boolean that indicates whether to include auxiliary tasks or not
        """
        self.files = sorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith(".pkl")])

        datum = self.pickle_load(self.files[0])

        self.img_size = (datum['img'].shape[0], datum['img'].shape[1])

        self.classes = datum['rock_classes']
        self.label_info = {}
        for i, name in enumerate(self.classes):
            self.label_info[i] = name

        self.transform = transform
        self.auxiliary = auxiliary

    @property
    def num_labels(self) -> int:
        """ Number of labels in the dataset
        """
        return len(self.label_info)

    @property
    def categories(self) -> List[str]:
        """ Name of all labels (object categories)
        """
        return self.classes

    @property
    def label_map(self) -> Dict[int, str]:
        """ Map from label num to name of label
        """
        return self.label_info

    def pickle_load(self, filepath: str) -> Any:
        """ Loads pickled data from a given path
        """
        with open(filepath, 'rb') as handle:
            loaded_data = pickle.load(handle)
        return loaded_data

    def pickle_save(self, filepath: str):
        """ Saves data to a given path as a .pkl file
        """
        with open(filepath, 'wb') as handle:
            pickle.dump(self, handle)

    def get_eval(self, idx: int) -> Tuple[List[Any], List[int]]:
        """ Gets the bounding box info of a sample for evaluation

        Args:
            idx: image index

        Returns:
            tuple containing:
                list of bounding box location and list of bounding box labels

        """
        d = self.pickle_load(self.files[idx])

        bbox_sizes = []
        bbox_labels = []

        for elem in (d['bboxes']):
            l, t, r, b = elem['box']
            w, h = r - l, b - t

            bbox_size = (l, t, w, h)
            bbox_label = elem['labels']
            bbox_sizes.append(bbox_size)
            bbox_labels.append(bbox_label)

        return bbox_sizes, bbox_labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """ Obtains a sample from a given index of the dataset

        Output types of dict items are:
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
        """

        # Load sample files
        d = self.pickle_load(self.files[idx])

        img = Image.fromarray(d['img'])
        img_id = idx

        htot, wtot = self.img_size

        bboxes, labels = [], []
        for elem in (d['bboxes']):
            l, t, r, b = elem['box']
            bbox_label = elem['labels']
            bbox_size = (l / wtot, t / htot, r / wtot, b / htot)
            bboxes.append(bbox_size)
            labels.append(bbox_label)

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        labels = torch.tensor(labels)
        mask = None

        # Auxiliary tasks part
        if self.auxiliary:
            scene_id = d['scene_id']
            depth = d['depth']
            max_depth = depth.max()

            depth = np.uint8(255 * depth / max_depth)
            depth = Image.fromarray(depth, 'L')

            normals = Image.fromarray(d['normals'])
            normals_mask = Image.fromarray(d['mask'])
        else:
            scene_id, depth, max_depth, normals, normals_mask = 0, 0, 0, 0, 0

        sample = {'img': img, 'img_id': img_id, 'size': (htot, wtot),
                  'bboxes': bboxes, 'labels': labels, 'bboxes_mask': mask,
                  'scene_id': scene_id, 'depth': depth, 'normals': normals, 'normals_mask': normals_mask,
                  'auxiliary': self.auxiliary}

        sample = self.transform(sample)

        if self.auxiliary:
            sample['depth'] = sample['depth'] * max_depth

        return sample
