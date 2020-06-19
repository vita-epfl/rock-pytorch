from typing import Dict, Optional

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt, patches as patches
from torchvision import transforms as transforms

import rock.ssd.encoder
from rock.model import losses


def write_scenes(t: torch.Tensor, save_path: str, log: bool = True) -> None:
    """ Writes scene predictions in sorted order to a given file
    """
    scene_types = ['basement', 'bathroom', 'bedroom', 'bookstore', 'cafe', 'classroom', 'computer_lab',
                   'conference_room', 'dinette', 'dining_room', 'exercise_room', 'foyer', 'furniture_store',
                   'home_office', 'home_storage', 'indoor_balcony', 'kitchen', 'laundry_room', 'living_room',
                   'office', 'office_kitchen', 'playroom', 'printer_room', 'reception_room', 'student_lounge',
                   'study', 'study_room']

    if log:
        t = torch.exp(t)

    t = t.flatten().tolist()

    sorted_list = sorted(list(zip(scene_types, t)), key=(lambda x: x[1]), reverse=True)

    with open(save_path, 'w') as f:
        for i, (scene, value) in enumerate(sorted_list, start=1):
            print('{:02d}. {}: {:.3f}'.format(i, scene, value), file=f)


def draw_normals(t: torch.Tensor, save_path: str) -> None:
    """Shows the network's predicted normals for an image
    """

    if t.dim() == 4:
        t = losses.normalize(t)
    elif t.dim() == 3:
        t = t.unsqueeze(dim=0)
        t = losses.normalize(t)
        t = t.squeeze()

    t = (t + 1) / 2
    torchvision.utils.save_image(t, fp=save_path)


def draw_depth(t: torch.Tensor, save_path: str, log: bool = True) -> None:
    """Shows the network's predicted depth for an image
    """
    if log:
        t = torch.exp(t)

    max_depth, _ = torch.max(t.reshape(t.shape[0], -1), dim=-1)

    for _ in range(t.dim() - 1):
        max_depth = max_depth.unsqueeze(dim=-1)

    t = t / max_depth

    # Inverse colors
    t = 1 - t
    torchvision.utils.save_image(t, fp=save_path)


def draw_predictions(img: torch.Tensor,
                     encoder: rock.ssd.encoder.Encoder,
                     ploc: torch.Tensor,
                     plabel: torch.Tensor,
                     idx: int,
                     label_map: Dict[int, str],
                     show: bool = True,
                     save_path: Optional[str] = None,
                     conf_threshold: float = 0.0) -> None:
    """Shows an input image and the predicted bounding boxes and confidence
    """
    if label_map is None:
        label_map = {}
    img_height, img_width, _ = img.shape

    dec = encoder.decode_batch(ploc, plabel)[idx]

    pred_boxes, labels, confs = dec

    labels = np.array(labels)
    labels_num = None

    if label_map:
        labels_num = np.array(labels)
        labels = [label_map.get(label) for label in labels]

    fig, ax = plt.subplots(1)

    # If nothing predicted, don't add anything
    if not pred_boxes.shape[0] == 0:
        # Get box sizes in pixel and convert to l,t,w,h for COCOeval
        boxes = pred_boxes.cpu() * torch.tensor([img_width, img_height, img_width, img_height])
        l, t, r, b = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx, cy, w, h = (l + r) / 2, (t + b) / 2, r - l, b - t
        cx, cy, w, h = np.array(cx), np.array(cy), np.array(w), np.array(h)

        bboxes = zip(cx, cy, w, h)

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        for (cx, cy, w, h), label, num_labels, conf in zip(bboxes, labels, labels_num, confs):
            if label == "background" or conf.item() < conf_threshold:
                continue

            color = colors[num_labels]
            bbox = patches.Rectangle((cx - 0.5 * w, cy - 0.5 * h), w, h,
                                     linewidth=3, edgecolor=color, facecolor='none')

            ax.add_patch(bbox)
            box_text = label + ' (' + str(round(conf.item() * 100, 1)) + '%)'
            plt.text(cx - 0.5 * w, cy - 0.5 * h, s=box_text,
                     color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})

    img = img.cpu()
    ax.imshow(img)

    if save_path:
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.axis('on')

    if show:
        plt.show()

    plt.close(fig)


def draw_transforms(img: torch.Tensor,
                    bboxes: torch.Tensor,
                    labels: torch.Tensor,
                    label_map: Dict[int, str],
                    show: bool = True,
                    save_path: str = None) -> None:
    """Draws transformed image
    """

    # Modified from original draw patches method
    # Suppose bboxes in fractional coordinates
    if label_map is None:
        label_map = {}
    img = np.array(img)
    labels = np.array(labels)

    bboxes = np.array(bboxes.numpy())

    labels_num = np.array(labels)
    labels = [label_map.get(label) for label in labels]

    cx, cy, w, h = bboxes[0, :], bboxes[1, :], bboxes[2, :], bboxes[3, :]

    htot, wtot, _ = img.shape
    cx *= wtot
    cy *= htot
    w *= wtot
    h *= htot

    bboxes = zip(cx, cy, w, h)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    fig, ax = plt.subplots(1)

    for (cx, cy, w, h), label, num_labels in zip(bboxes, labels, labels_num):
        if label == "background":
            continue

        color = colors[num_labels]

        bbox = patches.Rectangle((cx - 0.5 * w, cy - 0.5 * h), w, h,
                                 linewidth=3, edgecolor=color, facecolor='none')

        ax.add_patch(bbox)
        plt.text(cx - 0.5 * w, cy - 0.5 * h, s=label,
                 color='white', verticalalignment='top',
                 bbox={'color': color, 'pad': 0})

    ax.imshow(img)

    if save_path:
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.axis('on')

    if show:
        plt.show()

    plt.close(fig)


def inv_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse of the normalization that was done during pre-processing
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    return inv_normalize(tensor).permute(1, 2, 0)


def rock_label_map() -> Dict[int, str]:
    """ Mapping from label num to label name
    """
    label_map = {0: 'background', 1: 'bathtub', 2: 'bed', 3: 'bookshelf', 4: 'box', 5: 'chair', 6: 'counter', 7: 'desk',
                 8: 'door', 9: 'dresser', 10: 'garbage bin', 11: 'lamp', 12: 'monitor', 13: 'night stand', 14: 'pillow',
                 15: 'sink', 16: 'sofa', 17: 'table', 18: 'television', 19: 'toilet'}

    return label_map
