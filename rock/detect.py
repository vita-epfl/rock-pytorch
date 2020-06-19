import os
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.utils.data

import rock.ssd.prior_boxes
import rock.ssd.encoder
import rock.datasets.transforms
import rock.datasets.image_folder
import rock.model.network
import rock.utils.load
import rock.utils.draw


def object_detection(model_path: str,
                     image_folder_path: str = 'data/detection/images',
                     detection_output_path: str = 'data/detection/output',
                     scene_output_path: Optional[str] = None,
                     depth_output_path: Optional[str] = None,
                     normals_output_path: Optional[str] = None,
                     device: torch.device = torch.device("cuda"),
                     aux: bool = True,
                     aux_tasks: Tuple[str, ...] = ('scene', 'depth', 'normals'),
                     conf_threshold: float = 0.4,
                     throughput: bool = False,
                     verbose: bool = True) -> None:
    """ Loads a model and detects images at a given path
    """
    if detection_output_path:
        Path(detection_output_path).mkdir(parents=True, exist_ok=True)
    if scene_output_path:
        Path(scene_output_path).mkdir(parents=True, exist_ok=True)
    if depth_output_path:
        Path(depth_output_path).mkdir(parents=True, exist_ok=True)
    if normals_output_path:
        Path(normals_output_path).mkdir(parents=True, exist_ok=True)

    if verbose and not throughput:
        print("Running object detection with model: {}".format(model_path))

    if throughput:
        print("Calculating throughput disables saving detection output to folder")
    pboxes = rock.ssd.prior_boxes.pboxes_rock()
    encoder = rock.ssd.encoder.Encoder(pboxes)
    image_data = rock.datasets.image_folder.ImageFolder(image_folder_path)

    model = rock.model.network.rock_network(aux_tasks) if aux else rock.model.network.baseline_ssd()
    model = model.to(device)
    rock.utils.load.load_from_checkpoint(model_path, model, verbose=verbose)

    predict(model=model, dataset=image_data, encoder=encoder, device=device,
            conf_threshold=conf_threshold, detection_output_path=detection_output_path,
            scene_output_path=scene_output_path, depth_output_path=depth_output_path,
            normals_output_path=normals_output_path, aux=aux, aux_tasks=aux_tasks, throughput=throughput,
            verbose=verbose)

    if verbose and not throughput:
        print("Detections saved to: {}".format(detection_output_path))
        print("Done!")


def predict(model: torch.nn.Module,
            dataset: torch.utils.data.Dataset,
            encoder: rock.ssd.encoder.Encoder,
            detection_output_path: str,
            scene_output_path: str,
            depth_output_path: str,
            normals_output_path: str,
            device: torch.device,
            aux: bool,
            aux_tasks: Tuple[str, ...],
            conf_threshold: float,
            throughput: bool,
            verbose: bool) -> float:
    """ Performs object detection for a given model

    Returns the number of images evaluated per sec (forward pass) if show_images_per_sec is False, otherwise,
    prints the number of images evaluated per sec
    """
    model.eval()
    model.to(device)

    batch_size = 1 if throughput else 8
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

    total_images = len(dataset)
    total_time = 0

    for i, (imgs, filenames) in enumerate(loader):
        tic = time.time()
        with torch.no_grad():
            imgs = imgs.to(device)
            ploc, plabel, *aux_out = model(imgs)

            toc = time.time()
            total_time += (toc - tic)

            # Save images only if we are not checking the throughput
            if not throughput:
                for j in range(imgs.shape[0]):
                    save_path = os.path.join(detection_output_path, filenames[j])
                    rock.utils.draw.draw_predictions(img=rock.utils.draw.inv_norm(imgs[j]),
                                                     encoder=encoder, ploc=ploc, plabel=plabel, idx=j,
                                                     label_map=rock.utils.draw.rock_label_map(), show=False,
                                                     save_path=save_path, conf_threshold=conf_threshold)

                    if aux:
                        if 'scene' in aux_tasks and scene_output_path:
                            scene = aux_out[0]
                            scene_save_path = os.path.join(scene_output_path, filenames[j])
                            scene_save_path = os.path.splitext(scene_save_path)[0] + '.txt'
                            rock.utils.draw.write_scenes(scene[j], scene_save_path, log=True)

                        if 'depth' in aux_tasks and depth_output_path:
                            depth = aux_out[1]
                            depth_save_path = os.path.join(depth_output_path, filenames[j])
                            rock.utils.draw.draw_depth(depth[j], depth_save_path, log=True)

                        if 'normals' in aux_tasks and normals_output_path:
                            normals = aux_out[2]
                            normals_save_path = os.path.join(normals_output_path, filenames[j])
                            rock.utils.draw.draw_normals(normals[j], normals_save_path)

        if verbose or throughput:
            print("{}/{} images detected".format((i+1) * batch_size, total_images), end='\r')

    model.train()

    images_per_sec = total_images / total_time

    if throughput:
        print()
        print("Throughput: {:.2f} images/sec".format(images_per_sec))
    elif verbose:
        print("{}/{} images detected".format(total_images, total_images))

    return images_per_sec
