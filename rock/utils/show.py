import os
from pathlib import Path
from typing import Dict, List

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

import rock.ssd.encoder
from rock.utils.draw import draw_transforms, inv_norm, draw_predictions


def predict_grid(model: torch.nn.Module,
                 dataset: torch.utils.data.Dataset,
                 encoder: rock.ssd.encoder.Encoder,
                 label_map: Dict[int, str],
                 device: torch.device = torch.device("cuda"),
                 conf_threshold: float = 0.0) -> torch.Tensor:
    """ Obtains grid of batch images with ground truth boxes and the model's detections

    Args:
        model: network
        dataset: dataset on which to output grid
        encoder: encoder use to encode / decode the network's output
        label_map: dictionary mapping class id -> class name
        device: device to use to obtain grids (default: cuda)
        conf_threshold: the confidence threshold to show detections (default: 0.0)

    Returns:
        grid of images batch

    """
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
    dataiter = iter(dataloader)

    sample = next(dataiter)

    return _batch_image(sample, model, encoder, label_map, device, conf_threshold)


def all_predict_grids(model: torch.nn.Module,
                      dataset: torch.utils.data.Dataset,
                      encoder: rock.ssd.encoder.Encoder,
                      label_map: Dict[int, str],
                      device: torch.device = torch.device("cuda"),
                      conf_threshold: float = 0.0) -> List[torch.Tensor]:
    """ Obtains all grids of batch images with ground truth boxes and the model's detections

    Args:
        model: network
        dataset: dataset on which to output grid
        encoder: encoder use to encode / decode the network's output
        label_map: dictionary mapping class id -> class name
        device: device to use to obtain grids (default: cuda)
        conf_threshold: the confidence threshold to show detections (default: 0.0)

    Returns:
        all grids of images batch
    """
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    grids = []
    for i, sample in enumerate(dataloader):
        grids.append(_batch_image(sample, model, encoder, label_map, device, conf_threshold))

    return grids


def _batch_image(sample: Dict[str, torch.Tensor],
                 model: torch.nn.Module,
                 encoder: rock.ssd.encoder.Encoder,
                 label_map: Dict[int, str],
                 device: torch.device = torch.device("cuda"),
                 conf_threshold: float = 0.0) -> torch.Tensor:

    # Put model in eval mode
    batch_size = 8
    model.eval()
    model.to(device)

    imgs, bboxes, labels = sample['img'], sample['bboxes'], sample['labels']

    i = 0
    save_folder = 'data/tensorboard_images'
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    save_paths = []
    for img, bbox, label in zip(imgs, bboxes, labels):
        save_path = os.path.join(save_folder, 'gt_{}.png'.format(i))
        save_paths.append(save_path)
        draw_transforms(inv_norm(img), bbox, label, label_map=label_map, save_path=save_path, show=False)
        i += 1

    with torch.no_grad():
        inp = imgs.clone().to(device)
        ploc, plabel, *aux_out = model(inp)

        for i in range(batch_size):
            save_path = os.path.join(save_folder, 'pt_{}.png'.format(i))
            save_paths.append(save_path)
            draw_predictions(inv_norm(imgs[i]), encoder, ploc, plabel, i, label_map=label_map, save_path=save_path, show=False, conf_threshold=conf_threshold)

    # Put model back in training mode
    model.train()

    images = []

    for path in save_paths:
        img = Image.open(path)
        img = img.resize((320, 240))

        t = torchvision.transforms.ToTensor()(img)
        images.append(t)

    img_grid = torchvision.utils.make_grid(images, padding=5)

    return img_grid
