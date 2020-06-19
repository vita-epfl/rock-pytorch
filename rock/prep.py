import os
import pickle
from pathlib import Path
from typing import Tuple, Optional

import h5py
import numpy
import numpy as np
from PIL import Image
from scipy.io import loadmat


def prep_data(dataset_path: str,
              splits_path: str,
              normals_path: str,
              train_save_path: str,
              test_save_path: str,
              val_save_path: str,
              val_split_path: Optional[str] = None,
              verbose: bool = True) -> None:
    """ Prepares data (usually from given command-line arguments)
    """
    if verbose:
        if val_split_path:
            print("Train/val/test split")
        else:
            print("Train/test split")

        print("Beginning preprocessing...")

    dataset = NYUv2Preprocessing(dataset_path, splits_path, normals_path)

    if val_split_path:
        dataset.add_val(val_split_path)

    dataset.save(train_save_path, subset='train')
    if verbose:
        print("Saved train set to: {}".format(train_save_path))

    dataset.save(test_save_path, subset='test')
    if verbose:
        print("Saved test set to: {}".format(test_save_path))

    if val_split_path:
        dataset.save(val_save_path, subset='val')
        if verbose:
            print("Saved val set to: {}".format(val_save_path))

    if verbose:
        print("Done!")


class NYUv2Preprocessing(object):
    """Pre-processes the NYUv2 dataset
    Parses .mat files from the NYUv2 dataset, extracts necessary info
    and finds bounding boxes
    """

    def __init__(self, dataset_path: str, splits_path: str, normals_path: str) -> None:
        self.in_f = h5py.File(dataset_path, 'r')
        self.nyuv2 = {}

        for name, data in self.in_f.items():
            self.nyuv2[name] = data

        self.imgs, self.depths, self.labels, self.label_instances = self.__get_arrs()

        self.len = self.imgs.shape[0]

        self.scene_types = self.__read_mat_text(self.nyuv2['sceneTypes'])
        self.class_list = self.__read_mat_text(self.nyuv2['names'])
        self.scenes = self.__read_mat_text((self.nyuv2['scenes']))

        self.scene_ids, self.unique_scenes = self.__get_scene_ids()

        self.bboxes, self.rock_classes = self._get_all_bboxes()

        self.train_idx, self.test_idx = self._splits(splits_path)

        self.val = False
        self.val_idx = []

        self.masks, self.normals = get_surface_normals(normals_path)

    def save(self, path: str, subset: str = 'all') -> None:
        """Saves a specified subset of the data at a given folder path.
        
        Subset can be `train`, `test`, `val` or `all`.
        """
        Path(path).mkdir(parents=True, exist_ok=True)

        if subset == 'train':
            self._save_subset(path, self.train_idx, digits=4)
        elif subset == 'test':
            self._save_subset(path, self.test_idx, digits=4)
        elif subset == 'val':
            self._save_subset(path, self.val_idx, digits=4)
        elif subset == 'all':
            self._save_subset(path, range(self.len), digits=4)
        else:
            print("Couldn't find specified subset")

    def add_val(self, path: str) -> None:
        """Adds validation set using the path of a file containing the list of scenes part of the validation set
        """

        if not self.val:
            with open(path, 'r') as f:
                y = f.read().splitlines()

            for i, elem in enumerate(self.scenes):
                if elem in y:
                    self.val_idx.append(i)

            self.train_idx = [i for i in self.train_idx if i not in self.val_idx]
            self.val = True

    def _save_elem(self, path: str, idx: int) -> None:
        """ Save a specified data sample at a given path
        """
        d = {}
        d['img'] = self.imgs[idx]
        d['depth'] = self.depths[idx]
        d['labels'] = self.labels[idx]
        d['label_instance'] = self.label_instances[idx]
        d['scene_type'] = self.scene_types[idx]
        d['scene'] = self.scenes[idx]
        d['scene_id'] = self.scene_ids[idx]
        d['normals'] = self.normals[idx]
        d['mask'] = self.masks[idx]
        d['bboxes'] = self.bboxes[idx]
        d['rock_classes'] = self.rock_classes
        d['unique_scenes'] = self.unique_scenes

        with open(path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_subset(self, path, idx, digits=4):
        """ Save a specified subset of the data at a given path
        """
        for elem in idx:
            filename = str(elem).rjust(digits, '0') + '.pkl'
            file_path = os.path.join(path, filename)
            self._save_elem(file_path, elem)

    @staticmethod
    def _splits(splits_path):
        """ Splits the dataset into a test set and training set
        """
        splits = loadmat(splits_path)

        train_splits = splits['trainNdxs'] - 1
        test_splits = splits['testNdxs'] - 1

        train_idx = [elem.item() for elem in train_splits]
        test_idx = [elem.item() for elem in test_splits]

        return train_idx, test_idx

    @staticmethod
    def _transpose_3d_from_mat(data):
        """ Transposes for .mat array format to numpy array format
        """
        elem_list = [np.transpose(elem, (2, 1, 0)) for elem in data]
        elems = np.stack(elem_list, axis=0)
        return elems

    @staticmethod
    def _transpose_2d_from_mat(data):
        """ Transposes for .mat array format to numpy array format
        """
        elem_list = [np.transpose(elem, (1, 0)) for elem in data]
        elems = np.stack(elem_list, axis=0)
        return elems

    def __get_arrs(self):
        """ Gets the images, depths, labels and label_instances as numpy arrays
        """
        imgs = self._transpose_3d_from_mat(self.nyuv2['images'])
        depths = self._transpose_2d_from_mat(self.nyuv2['depths'])
        labels = self._transpose_2d_from_mat(self.nyuv2['labels'])
        label_instances = self._transpose_2d_from_mat(self.nyuv2['instances'])

        return imgs, depths, labels, label_instances

    def __read_mat_text(self, h5_dataset):
        """ Reads text from a .mat file
        """
        item_list = [u''.join(chr(c.item()) for c in self.in_f[obj_ref]) for obj_ref in (h5_dataset[0, :])]
        return item_list

    def __get_scene_ids(self):
        """ Obtains the scene ids for each sample
        """
        unique_scenes = sorted(list(set(self.scene_types)))

        scene_type_to_id = {}
        for i, scene in enumerate(unique_scenes):
            scene_type_to_id[scene] = i

        scene_ids = [scene_type_to_id[scene_type] for scene_type in self.scene_types]

        return scene_ids, unique_scenes

    @staticmethod
    def _get_rock_class(names_to_ids):
        """ Obtains the object classes and ids for the ROCK paper
        """

        rock_class_names = ['bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk',
                            'door', 'dresser', 'garbage bin', 'lamp', 'monitor', 'night stand',
                            'pillow', 'sink', 'sofa', 'table', 'television', 'toilet']

        rock_class_ids = [names_to_ids[name] for name in rock_class_names]

        return rock_class_names, rock_class_ids

    @staticmethod
    def _get_image_instances(labels, label_instances, class_ids):
        """ Obtains the object instances
        """
        # Keep only the objects from the indicated classes
        masks = np.isin(labels, class_ids)
        instances_masked = label_instances * masks
        labels_masked = labels * masks

        # Create new array of the same shape as the original instance array
        arr = np.zeros(labels.shape)

        # For each image
        for i in range(labels.shape[0]):
            count = 0

            # Get the classes of that image
            classes = np.unique(labels_masked[i, :, :])
            classes = classes[classes != i]

            # Map each instance of each class to a unique instance number
            for elem in classes:
                class_instances = instances_masked[i, :, :] * (labels_masked[i, :, :] == elem)

                for j in range(1, class_instances.max() + 1):
                    count += 1
                    arr[i, :, :][class_instances == j] = count

        return arr.astype(int)

    # noinspection PyDictCreation
    @staticmethod
    def _bbox(instances, labels, ids_to_names, class_names):
        """Obtains the bounding boxes for an image
        """

        bbox_list = []
        for (i, instance) in enumerate(instances):
            img_bbox_list = []
            # noinspection PyDictCreation
            for j in range(1, instance.max() + 1):
                a = np.where(instance == j)
                bbox = {}
                bbox['box'] = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])  # x1, y1, x2, y2
                old_label = labels[i, a[0][0], a[1][0]]
                bbox['label_name'] = ids_to_names[old_label]

                # Remap to new indices (from 1 to 20, in alphabetical order)
                bbox['labels'] = class_names.index(bbox['label_name'])
                img_bbox_list.append(bbox)
            bbox_list.append(img_bbox_list)

        return bbox_list

    def _get_all_bboxes(self):
        """ Obtains all bounding boxes for the specified rock classes
        """

        labels, label_instances, class_list = self.labels, self.label_instances, self.class_list
        names_to_ids = {}
        ids_to_names = {}
        for i, name in enumerate(class_list, start=1):
            names_to_ids[name] = i
            ids_to_names[i] = name

        rock_class_names, rock_class_ids = self._get_rock_class(names_to_ids)

        # Add background class as the first index
        rock_class_names.insert(0, "background")

        rock_instances = self._get_image_instances(labels, label_instances, rock_class_ids)

        bboxes = self._bbox(rock_instances, labels, ids_to_names, rock_class_names)

        return bboxes, rock_class_names


def get_surface_normals(path: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Obtains arrays of surface normals and normals mask arrays from input image

    Args:
        path (str): path of the folder containing folders of normals and masks

    Returns:
        (tuple): tuple containing:
            masks (numpy.ndarray): array of image masks
            normals (numpy.ndarray): list of normals
    """

    masks_path = os.path.join(path, "masks")
    normals_path = os.path.join(path, "normals")

    masks_files = sorted([os.path.join(masks_path, file) for file in os.listdir(masks_path) if file.endswith(".png")])
    normals_files = sorted(
        [os.path.join(normals_path, file) for file in os.listdir(normals_path) if file.endswith(".png")])

    masks = np.stack([np.array(Image.open(file)) for file in masks_files], axis=0)
    normals = np.stack([(np.array(Image.open(file))) for file in normals_files], axis=0)

    return masks, normals



