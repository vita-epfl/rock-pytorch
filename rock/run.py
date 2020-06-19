import argparse
from typing import Union

import torch


def int_or_none(value: str) -> Union[None, int]:
    if value == 'None':
        return None
    return int(value)


def cli() -> argparse.Namespace:
    """ Command line interface
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Different parsers for main actions', dest='command')

    prep_epilog = 'Note: If no val_split_path is provided, this command creates a train/test set. ' \
                  'Otherwise, this command creates a train/val/test set, ' \
                  'with the val set extracted from the test set. ' \
                  'It is recommended to change the default train_save_path and test_save_path when adding a val set.'
    prep_parser = subparsers.add_parser("prep",
                                        help='preprocess the NYUv2 dataset',
                                        epilog=prep_epilog)

    train_parser = subparsers.add_parser("train",
                                         help='train the ROCK network')

    eval_parser = subparsers.add_parser("eval",
                                        help='evaluate a trained ROCK network using COCOeval')

    image_folder_parser = subparsers.add_parser("create_image_folder",
                                                help="creates an image folder from the preprocessed NYUv2 dataset")

    detect_parser = subparsers.add_parser("detect",
                                          help='detect objects using a trained network')

    # Arguments for preprocessing
    prep_parser.add_argument('--dataset_path',
                             help='path to the NYUv2 dataset (default: %(default)s)',
                             default='data/nyu_depth_v2_labeled.mat')
    prep_parser.add_argument('--splits_path',
                             help='path to the NYUv2 official splits (default: %(default)s)',
                             default='data/splits.mat')
    prep_parser.add_argument('--normals_path',
                             help='path to the folder containing the normals and normals masks (default: %(default)s)',
                             default='data/normals_gt')
    prep_parser.add_argument('--val_split_path',
                             help='path containing the samples to be used for validation. '
                                  'No validation data if no path is provided (default: %(default)s)',
                             default=None)
    prep_parser.add_argument('--train_save_path',
                             help='path where the train data will be saved (default: %(default)s)',
                             default='data/train_test/nyuv2_train')
    prep_parser.add_argument('--test_save_path',
                             help='path where the test data will be saved (default: %(default)s)',
                             default='data/train_test/nyuv2_test')
    prep_parser.add_argument('--val_save_path',
                             help='path where the val data will be saved '
                                  '(if an argument for --val_split_path is provided) (default: %(default)s)',
                             default='data/train_val_test/nyuv2_val')
    prep_parser.add_argument('--no_verbose',
                             help='disable verbose', action='store_true')

    # Arguments for training
    train_parser.add_argument('--train_path',
                              help='path to the training data (default: %(default)s)',
                              default='data/train_test/nyuv2_train')
    train_parser.add_argument('--val_path',
                              help='path to the validation data (default: %(default)s)',
                              default=None)
    train_parser.add_argument('--device',
                              help='gpu used for training (type: %(type)s) (default: %(default)s)',
                              type=torch.device, default='cuda')
    train_parser.add_argument('--num_iters',
                              help='number of iterations (type: %(type)s) (default: %(default)s)',
                              type=int, default=40_000)
    train_parser.add_argument('--lr',
                              help='learning rate (type: %(type)s) (default: %(default)s)',
                              type=float, default=5e-5)
    train_parser.add_argument('--weight_decay',
                              help='weight decay for optimizer (type: %(type)s) (default: %(default)s)',
                              type=float, default=2e-3)
    train_parser.add_argument('--scheduler_milestones',
                              help='iteration milestones at which the learning rate is decreased '
                                   '(type: %(type)s) (default: %(default)s)',
                              type=int, default=[30_000, ], nargs='+')
    train_parser.add_argument('--scheduler_gamma',
                              help='gamma value for the scheduler, by which the learning rate is multiplied '
                                   'at each milestone (type: %(type)s) (default: %(default)s)',
                              type=float, default=0.1)
    train_parser.add_argument('--force_crops',
                              help='crop all training images during data augmentation, '
                                   'instead of leaving some images uncropped',
                              action='store_true')
    train_parser.add_argument('--no_rock',
                              help='remove rock block from model '
                                   '(obtains a baseline single shot detector, no auxiliary tasks)',
                              action='store_true')
    train_parser.add_argument('--aux_tasks',
                              help='list of auxiliary tasks to train on (type: %(type)s) (default: %(default)s)',
                              type=str, default=['scene', 'depth', 'normals'], nargs='*')
    train_parser.add_argument('--use_all_priors_conf_loss',
                              help='switches to a loss taking into account all negative examples (all priors) instead '
                                   'of just the top negative examples for the confidence loss.',
                              action='store_true')
    train_parser.add_argument('--writer_path',
                              help='path to the folder where the tensorboard runs will be stored '
                                   '(i.e. data/runs/rock) (default: %(default)s)',
                              default=None)
    train_parser.add_argument('--save_path',
                              help='path to the folder where the model weights will be saved '
                                   '(i.e. models/rock) (default: %(default)s)',
                              default=None)
    train_parser.add_argument('--checkpoint_path',
                              help='path to the folder where a trained model is saved '
                                   '(i.e. models/rock). If provided, training will be resumed on that model '
                                   '(default: %(default)s)',
                              default=None)
    train_parser.add_argument('--save_best_on_val',
                              help='saves the model with the best mAP on val data',
                              action='store_true')
    train_parser.add_argument('--val_eval_freq',
                              help='frequency at which the model is evaluated on the validation data, in epochs. '
                                   'If None, model is never evaluated (type: %(type)s) (default: %(default)s)',
                              type=int_or_none, default=10)
    train_parser.add_argument('--train_eval_freq',
                              help='frequency at which the model is evaluated on the training data, in epochs. '
                                   'If None, model is never evaluated (type: %(type)s) (default: %(default)s)',
                              type=int_or_none, default=50)
    train_parser.add_argument('--image_to_tb_freq',
                              help='frequency at which an image grid is added to Tensorboard, in epochs. '
                                   'If None, no image grid is added (type: %(type)s) (default: %(default)s)',
                              type=int_or_none, default=20)
    train_parser.add_argument('--model_save_freq',
                              help='frequency at which the model is saved, in epochs. '
                                   'If None, no model is saved at a specific epoch number '
                                   '(but can still be saved when training is finished or if --save_best_on_val is set) '
                                   '(type: %(type)s) (default: %(default)s)',
                              type=int_or_none, default=None)
    train_parser.add_argument('--no_verbose',
                              help='disable verbose', action='store_true')

    # Arguments for evaluation
    eval_parser.add_argument('model_path',
                             help='path containing the model weights')
    eval_parser.add_argument('--test_path',
                             help='path to the folder containing the test data on which to run the evaluation '
                                  '(default: %(default)s)',
                             default='data/train_test/nyuv2_test')
    eval_parser.add_argument('--device',
                             help='gpu used for evaluating (type: %(type)s) (default: %(default)s)',
                             type=torch.device, default='cuda')
    eval_parser.add_argument('--no_rock',
                             help='remove rock block from model '
                                  '(obtains a baseline single shot detector, no auxiliary tasks)',
                             action='store_true')
    eval_parser.add_argument('--aux_tasks',
                             help='list of auxiliary tasks to train on (type: %(type)s) (default: %(default)s)',
                             type=str, default=['scene', 'depth', 'normals'], nargs='*')
    eval_parser.add_argument('--show_all_cats',
                             help='show the mAP for all categories', action='store_true')
    eval_parser.add_argument('--no_verbose',
                             help='disable verbose', action='store_true')

    # Arguments for image folder creation
    image_folder_parser.add_argument('--data_path',
                                     help='path to the folder containing the images to extract (default: %(default)s)',
                                     default='data/train_test/nyuv2_test')
    image_folder_parser.add_argument('--save_path',
                                     help='path to the folder in which to save the new images (default: %(default)s)',
                                     default='data/detection/images')
    image_folder_parser.add_argument('--no_verbose',
                                     help='disable verbose', action='store_true')

    # Arguments for object detection
    detect_parser.add_argument('model_path',
                               help='path containing the model weights')
    detect_parser.add_argument('--image_path',
                               help='path to the folder in which the images are saved (default: %(default)s)',
                               default='data/detection/images')
    detect_parser.add_argument('--detection_output_path',
                               help='path to the folder in which the object detections are saved (default: %(default)s)',
                               default='data/detection/output')
    detect_parser.add_argument('--scene_output_path',
                               help='path to the folder where the scene predictions will be saved. '
                                    'Only works if the model contains a ROCK block (default: %(default)s)',
                               default=None)
    detect_parser.add_argument('--depth_output_path',
                               help='path to the folder where the depth predictions will be saved. '
                                    'Only works if the model contains a ROCK block (default: %(default)s)',
                               default=None)
    detect_parser.add_argument('--normals_output_path',
                               help='path to the folder where the surface normals predictions will be saved. '
                                    'Only works if the model contains a ROCK block (default: %(default)s)',
                               default=None)
    detect_parser.add_argument('--device',
                               help='device used for object detection (type: %(type)s) (default: %(default)s)',
                               type=torch.device, default='cuda')
    detect_parser.add_argument('--no_rock',
                               help='remove rock block from model '
                                    '(obtains a baseline single shot detector, no auxiliary tasks)',
                               action='store_true')
    detect_parser.add_argument('--aux_tasks',
                               help='list of auxiliary tasks to train on (type: %(type)s) (default: %(default)s)',
                               type=str, default=['scene', 'depth', 'normals'], nargs='*')
    detect_parser.add_argument('--conf_threshold',
                               help='only show objects above a certain confidence threshold '
                                    '(type: %(type)s) (default: %(default)s)',
                               type=float, default=0.4)
    detect_parser.add_argument('--get_throughput',
                               help='shows the throughput (images/sec) of the model (forward pass only). '
                                    'This disables saving the results of the object detection to a folder',
                               action='store_true')
    detect_parser.add_argument('--no_verbose',
                               help='disable verbose', action='store_true')

    args = parser.parse_args()

    return args


def disable_rock_for_empty_aux_tasks(args: argparse.Namespace) -> argparse.Namespace:
    """ Disable the rock block if no aux tasks are given, and transform aux_tasks to a tuple
    """
    args.aux_tasks = tuple(args.aux_tasks)
    if not args.aux_tasks:
        args.no_rock_block = True

    return args


def main() -> None:
    """ Parses the command-line arguments and calls the appropriate function
    """
    args = cli()

    if args.command == 'prep':
        from rock.prep import prep_data
        prep_data(dataset_path=args.dataset_path, splits_path=args.splits_path, normals_path=args.normals_path,
                  train_save_path=args.train_save_path, test_save_path=args.test_save_path,
                  val_save_path=args.val_save_path, val_split_path=args.val_split_path, verbose=not args.no_verbose)

    if args.command == 'train':
        from rock.trainer import train
        args = disable_rock_for_empty_aux_tasks(args)
        train(train_path=args.train_path, val_path=args.val_path, device=args.device, num_iters=args.num_iters,
              lr=args.lr, weight_decay=args.weight_decay, scheduler_milestones=args.scheduler_milestones,
              scheduler_gamma=args.scheduler_gamma, forced_crops=args.force_crops,
              aux=not args.no_rock, aux_tasks=args.aux_tasks, use_all_priors_conf_loss=args.use_all_priors_conf_loss,
              writer_path=args.writer_path, save_path=args.save_path, checkpoint_path=args.checkpoint_path,
              save_best_on_val=args.save_best_on_val, val_eval_freq=args.val_eval_freq,
              train_eval_freq=args.train_eval_freq, image_to_tb_freq=args.image_to_tb_freq,
              model_save_freq=args.model_save_freq, verbose=not args.no_verbose)

    if args.command == 'eval':
        from rock.eval import evaluate_model
        args = disable_rock_for_empty_aux_tasks(args)
        evaluate_model(model_path=args.model_path, test_path=args.test_path, device=args.device,
                       aux=not args.no_rock, aux_tasks=args.aux_tasks, show_all_cats=args.show_all_cats,
                       verbose=not args.no_verbose)

    if args.command == 'create_image_folder':
        from rock.datasets.image_folder import extract_image_and_save_to_folder
        extract_image_and_save_to_folder(data_folder_path=args.data_path, save_folder_path=args.save_path,
                                         verbose=not args.no_verbose)

    if args.command == 'detect':
        from rock.detect import object_detection
        args = disable_rock_for_empty_aux_tasks(args)
        object_detection(model_path=args.model_path, image_folder_path=args.image_path,
                         detection_output_path=args.detection_output_path, scene_output_path=args.scene_output_path,
                         depth_output_path=args.depth_output_path, normals_output_path=args.normals_output_path,
                         device=args.device, aux=not args.no_rock, aux_tasks=args.aux_tasks,
                         conf_threshold=args.conf_threshold, throughput=args.get_throughput,
                         verbose=not args.no_verbose)


if __name__ == '__main__':
    main()
