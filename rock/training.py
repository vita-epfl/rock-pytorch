from typing import Optional, Dict, Any, Tuple

import torch
import torch.optim
import torch.utils.data.dataloader
import torch.utils.tensorboard.writer


def train_loop(model: torch.nn.Module,
               loss_func: torch.nn.Module,
               dataloader: torch.utils.data.dataloader.DataLoader,
               epoch: int,
               iteration: int,
               optimizer: Any,  # from torch.optim (i.e. Adam, SGD, ...)
               scheduler: Any,  # from torch.optim.lr_scheduler (i.e. MultiStepLR)
               writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None,
               device: torch.device = torch.device("cuda")) -> Tuple[float, int]:
    """ Training loop
    Implements the forward + backward pass for each sample and obtains the loss on the training data
    """
    model.train()

    train_running_loss = 0.0
    train_loss_dict = None if writer is None else loss_dict()

    for sample in dataloader:
        optimizer.zero_grad()

        loss_sample = forward_pass(model, sample, device)

        loss = loss_func(loss_sample, train_loss_dict)
        train_running_loss += loss.item()
        loss.backward()

        # If we want gradient clipping:
        # clipping_value = 1  # arbitrary value of your choosing
        # torch.nn.utils.clip_grad_value_(model.parameters(), clipping_value)
        optimizer.step()

        scheduler.step()
        iteration += 1

    train_loss = train_running_loss / len(dataloader)

    if writer is not None:
        writer.add_scalar('train loss', train_loss, epoch)

        for key, value in train_loss_dict.items():
            writer.add_scalar(key + '_train', value / len(dataloader), epoch)

    return train_loss, iteration


def val_loop(model: torch.nn.Module,
             loss_func: torch.nn.Module,
             dataloader: torch.utils.data.dataloader.DataLoader,
             epoch: int,
             writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None,
             device: torch.device = torch.device("cuda"),
             dataset_type: str = 'val') -> float:
    """ Validation loop (no backprop)
    Obtains the loss on the validation (or test) data
    """
    model.eval()

    val_running_loss = 0.0
    val_loss_dict = loss_dict()

    # Get val data loss
    for sample in dataloader:
        with torch.no_grad():
            loss_sample = forward_pass(model, sample, device)

            loss = loss_func(loss_sample, val_loss_dict)

            val_running_loss += loss.item()

    val_loss = val_running_loss / len(dataloader)

    if writer is not None:
        writer.add_scalar('val loss'.format(dataset_type), val_loss, epoch)

        for key, value in val_loss_dict.items():
            writer.add_scalar('{}_{}'.format(key, dataset_type), value / len(dataloader), epoch)

    return val_loss


def forward_pass(model: torch.nn.Module,
                 sample: Dict[str, torch.Tensor],
                 device: torch.device) -> Dict[str, torch.Tensor]:
    """ Forward pass of the network for a given sample
    """
    img = sample['img'].to(device)
    bboxes = sample['bboxes'].to(device)
    labels = sample['labels'].to(device)
    bboxes_mask = sample['bboxes_mask'].to(device)

    # Forward pass
    ploc, plabel, *aux_out = model(img)

    # Create the loss sample
    loss_sample = {}
    loss_sample['ploc'] = ploc
    loss_sample['plabel'] = plabel
    loss_sample['bboxes'] = bboxes
    loss_sample['labels'] = labels
    loss_sample['bboxes_mask'] = bboxes_mask

    # Check if we should include auxiliary
    if sample['auxiliary'].all() and aux_out is not None:
        loss_sample['scene_gt'] = sample['scene_id'].to(device)
        loss_sample['scene_pred'] = aux_out[0]

        loss_sample['depth_gt'] = sample['depth'].to(device)
        loss_sample['depth_pred'] = aux_out[1]

        loss_sample['normals_gt'] = sample['normals'].to(device)
        loss_sample['normals_mask'] = sample['normals_mask'].to(device)
        loss_sample['normals_pred'] = aux_out[2]

    return loss_sample


def loss_dict() -> Dict[str, float]:
    """ Initialize loss dictionary for network sub-tasks
    """

    # Add z in front of each key name so they appear on the bottom (and together) in tensorboard
    d = {'z_normals_loss': 0.0, 'z_scene_loss': 0.0, 'z_depth_loss': 0.0, 'z_conf_loss': 0.0, 'z_loc_loss': 0.0}
    return d
