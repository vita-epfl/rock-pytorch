import math
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union, Iterable

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import rock.datasets.nyu_depth_v2
import rock.model.network
import rock.eval
import rock.utils.load
import rock.utils.show
import rock.utils.hide_print
import rock.ssd.encoder
import rock.ssd.prior_boxes
import rock.datasets.transforms
import rock.model.losses
import rock.training


def train(train_path: str,
          val_path: Optional[str] = None,
          device: torch.device = torch.device("cuda"),
          num_iters: int = 40_000,
          lr: float = 5e-5,
          weight_decay: float = 2e-3,
          scheduler_milestones: Iterable[int] = (30_000, ),
          scheduler_gamma: float = 0.1,
          forced_crops: bool = False,
          aux: bool = True,
          aux_tasks: Tuple[str, ...] = ('scene', 'depth', 'normals'),
          use_all_priors_conf_loss: bool = False,
          writer_path: Optional[str] = None,
          save_path: Optional[str] = None,
          checkpoint_path: Optional[str] = None,
          save_best_on_val: bool = False,
          val_eval_freq: Union[int, None] = 10,
          train_eval_freq: Union[int, None] = 50,
          image_to_tb_freq: Union[int, None] = 20,
          model_save_freq: Union[int, None] = None,
          verbose: bool = True) -> None:
    # Initialize dataset and model
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_path, '')

    if writer_path:
        writer_path = os.path.join(writer_path, '')

    pboxes = rock.ssd.prior_boxes.pboxes_rock()
    encoder = rock.ssd.encoder.Encoder(pboxes)

    train_trans = rock.datasets.transforms.Transformer(pboxes, (480, 640), train=True, forced_crops=forced_crops)
    test_trans = rock.datasets.transforms.Transformer(pboxes, (480, 640), train=False)

    train_data = rock.datasets.nyu_depth_v2.NYUv2Detection(train_path, transform=train_trans, auxiliary=aux)
    train_eval_data = rock.datasets.nyu_depth_v2.NYUv2Detection(train_path, transform=test_trans, auxiliary=aux)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2, drop_last=True)

    if val_path:
        val_data = rock.datasets.nyu_depth_v2.NYUv2Detection(val_path, transform=test_trans, auxiliary=aux)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, num_workers=2, drop_last=True)
    else:
        val_data = None
        val_loader = None

    label_map = train_data.label_map

    train_len = len(train_loader)

    epochs = math.ceil(num_iters / train_len)
    model = rock.model.network.rock_network(aux_tasks) if aux else rock.model.network.baseline_ssd()
    model = model.to(device)
    loss_func = rock.model.losses.Loss(pboxes,
                                       auxiliary=aux,
                                       aux_tasks=aux_tasks,
                                       use_all_priors_conf_loss=use_all_priors_conf_loss).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)

    writer = SummaryWriter(writer_path) if writer_path is not None else None

    total_time = 0
    iteration = 0
    total_iters = 0
    start_epoch = 0
    best_ap = -1.0
    ap1, ap2 = 0.0, 0.0

    # Loading model
    if checkpoint_path:
        start_epoch, iteration = rock.utils.load.load_from_checkpoint(checkpoint_path,
                                                                      model, optimizer, scheduler, verbose)

        if verbose:
            print("Resuming training of model stored at: {}".format(checkpoint_path))
            print(scheduler.state_dict())

    if verbose:
        if aux:
            print("ROCK block enabled")
        else:
            print("ROCK block disabled")

        if val_path:
            print("Training on {}, evaluating on {}".format(train_path, val_path))
        else:
            print("Training on {}, no val set".format(train_path))

    print("Training the model for {} epochs ({} iters) \n".format(epochs, num_iters))

    ###############
    # Training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        start_epoch_time = time.time()

        train_loss, iteration = rock.training.train_loop(model, loss_func, train_loader, epoch, iteration,
                                                         optimizer, scheduler, writer=writer, device=device)

        if val_loader:
            val_loss = rock.training.val_loop(model, loss_func, val_loader, epoch, writer=writer, device=device)
        else:
            val_loss = None

        out_str = '[Epoch {}] train loss: {:.4f}'.format(epoch, train_loss)
        if val_loss:
            out_str += ' / val loss: {:.4f}'.format(val_loss)

        if verbose:
            print(out_str)
        else:
            if ap1 > 0.0:
                out_str += ' / latest val mAP[0.50:0.95]: {:.4f}'.format(ap1)
            print(out_str, end='\r')

        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        ########################
        # Evaluate and save model

        # Val data eval
        if val_data and val_eval_freq and epoch % val_eval_freq == 0:
            if verbose:
                print()
                print("[Val] Epoch {} eval".format(epoch))
                ap1, ap2 = rock.eval.evaluate(model, val_data, encoder, device)
            else:
                with rock.utils.hide_print.HiddenPrints():
                    ap1, ap2 = rock.eval.evaluate(model, val_data, encoder, device)

            if writer:
                writer.add_scalar('val mAP[0.50:0.95]', ap1, epoch)
                writer.add_scalar('val mAP[0.50]', ap2, epoch)

            if verbose:
                print('val mAP[0.50:0.95] = {:.4f}'.format(ap1))
                print('val mAP[0.50] = {:.4f}'.format(ap2))
                print()

            # Save model if it has the best ap
            if save_path and save_best_on_val:
                if (ap1 + ap2) > best_ap:
                    best_ap = ap1 + ap2
                    obj = {'epoch': epoch+1, 'iteration': iteration, 'optimizer': optimizer.state_dict(),
                           'scheduler': scheduler.state_dict(),
                           'model': model.state_dict()}
                    torch.save(obj, '{}best_model.pt'.format(save_path))

                    if verbose:
                        print("new best mAP, saved model at epoch {}".format(epoch))

        # Train data eval
        if train_eval_freq and epoch % train_eval_freq == 0:
            if verbose:
                print()
                print("[Train] Epoch {} eval".format(epoch))
                train_ap1, train_ap2 = rock.eval.evaluate(model, train_eval_data, encoder, device)
            else:
                with rock.utils.hide_print.HiddenPrints():
                    train_ap1, train_ap2 = rock.eval.evaluate(model, train_eval_data, encoder, device)

            if writer:
                writer.add_scalar('train mAP[0.50:0.95]', train_ap1, epoch)
                writer.add_scalar('train mAP[0.50]', train_ap2, epoch)

            if verbose:
                print('train mAP[0.50:0.95] = {:.4f}'.format(train_ap1))
                print('train mAP[0.50] = {:.4f}'.format(train_ap2))
                print()

        # Upload images to tensorboard
        if writer and image_to_tb_freq and epoch % image_to_tb_freq == 0:
            train_eval_grid = rock.utils.show.predict_grid(model, train_eval_data, encoder, label_map)
            writer.add_image("train_grid", train_eval_grid, epoch)

            if val_data:
                val_grid = rock.utils.show.predict_grid(model, val_data, encoder, label_map)
                writer.add_image("val_grid", val_grid, epoch)

            train_grid = rock.utils.show.predict_grid(model, train_data, encoder, label_map)
            writer.add_image("train_grid (with crops)", train_grid, epoch)

        # Save model
        if save_path and model_save_freq and epoch % model_save_freq == 0:
            obj = {'epoch': epoch+1, 'iteration': iteration, 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'model': model.state_dict()}
            torch.save(obj, '{}epoch_{}.pt'.format(save_path, epoch))
            if verbose:
                print("saved model at epoch {}".format(epoch))

        total_iters = iteration

    # End of training
    if save_path:
        obj = {'epoch': start_epoch + epochs, 'iteration': total_iters, 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(),
               'model': model.state_dict()}
        torch.save(obj, '{}final_model.pt'.format(save_path))
        print()
        print("saved final model")

    if val_data:
        print("Final model eval")

        if verbose:
            ap1, ap2 = rock.eval.evaluate(model, val_data, encoder, device, show_all_cats=True)
        else:
            with rock.utils.hide_print.HiddenPrints():
                ap1, ap2 = rock.eval.evaluate(model, val_data, encoder, device, show_all_cats=True)

        print('val mAP[0.50:0.95] = {:.4f}'.format(ap1))
        print('val mAP[0.50] = {:.4f}'.format(ap2))
        print()

    print('Total training time: {}'.format(total_time))

    if writer:
        writer.close()
