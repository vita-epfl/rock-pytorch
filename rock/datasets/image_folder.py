import os
import pickle
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import torch
from PIL import Image
from torch.utils import data as data
from torchvision import transforms


def extract_image_and_save_to_folder(data_folder_path: str,
                                     save_folder_path: str,
                                     verbose: bool = True) -> None:
    """ Extracts images from a folder of .pkl files and saves it to another folder
    """
    def pickle_load(fp: str) -> Any:
        """ Loads pickled data from a given path
        """
        with open(fp, 'rb') as handle:
            loaded_data = pickle.load(handle)
        return loaded_data

    Path(save_folder_path).mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Extracting images...")

    files = sorted([os.path.join(data_folder_path, file) for file in os.listdir(data_folder_path) if file.endswith(".pkl")])

    image_count = 0
    total_images = len(files)

    for filepath in files:
        filename = os.path.basename(filepath).replace('.pkl', '.png')
        save_path = os.path.join(save_folder_path, filename)

        d = pickle_load(filepath)
        img = Image.fromarray(d['img'])
        img.save(fp=save_path)

        image_count += 1
        if verbose:
            print("{}/{} images extracted".format(image_count, total_images), end='\r')

    if verbose:
        print()
        print("Done!")


class ImageFolder(data.Dataset):
    """Datareader for a folder of images
    """

    def __init__(self, path: str) -> None:
        """
        Args:
            path: path to the folder containing images (either PNG or JPG images)
        """
        self.files = sorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith(".jpg") or file.endswith(".png")])

        self.size = (480, 640)
        self.aspect_ratio = self.size[1] / self.size[0]

        self.img_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """ Obtains the image and filename at a given index
        """
        # Get file name
        filename = os.path.basename(self.files[idx])

        # Load image and remove alpha color channel if existing
        img = Image.open(self.files[idx]).convert("RGB")

        # Find largest crop
        if self.aspect_ratio < 1:
            crop_w = np.min([img.height * self.aspect_ratio, img.width])
            crop_h = crop_w / self.aspect_ratio
        else:
            crop_h = np.min([img.width / self.aspect_ratio, img.height])
            crop_w = crop_h * self.aspect_ratio

        img = transforms.CenterCrop(size=(crop_h, crop_w))(img)
        img = self.img_trans(img)

        return img, filename
