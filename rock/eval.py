import json
import os
from pathlib import Path
from typing import Tuple, Dict, List, Any, Union

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

import rock.datasets.nyu_depth_v2
import rock.ssd.encoder
import rock.ssd.prior_boxes
import rock.datasets.transforms
import rock.model.network
import rock.utils.load
import rock.utils.hide_print


def evaluate_model(model_path: str,
                   test_path: str = 'data/train_test/nyuv2_test',
                   device: torch.device = torch.device("cuda"),
                   aux: bool = True,
                   aux_tasks: Tuple[str, ...] = ('scene', 'depth', 'normals'),
                   coco_json_save_path: str = 'data/eval/',
                   show_all_cats: bool = False,
                   verbose: bool = True) -> None:
    """ Loads a model and evaluates it
    """
    if verbose:
        print("Evaluating model: {}".format(model_path))

    pboxes = rock.ssd.prior_boxes.pboxes_rock()
    encoder = rock.ssd.encoder.Encoder(pboxes)
    test_trans = rock.datasets.transforms.Transformer(pboxes, (480, 640), train=False)
    test_data = rock.datasets.nyu_depth_v2.NYUv2Detection(test_path, transform=test_trans, auxiliary=aux)

    model = rock.model.network.rock_network(aux_tasks) if aux else rock.model.network.baseline_ssd()
    model = model.to(device)
    rock.utils.load.load_from_checkpoint(model_path, model, verbose=verbose)

    gt_path = os.path.join(coco_json_save_path, 'gt_box.json')
    dt_path = os.path.join(coco_json_save_path, 'pred_box.json')

    if verbose:
        ap1, ap2 = evaluate(model, test_data, encoder, device, gt_path=gt_path, dt_path=dt_path,
                            show_all_cats=show_all_cats)
    else:
        with rock.utils.hide_print.HiddenPrints():
            ap1, ap2 = evaluate(model, test_data, encoder, device, gt_path=gt_path, dt_path=dt_path,
                                show_all_cats=show_all_cats)

    print('val mAP[0.50:0.95] = {:.4f}'.format(ap1))
    print('val mAP[0.50] = {:.4f}'.format(ap2))


def evaluate(model: torch.nn.Module,
             dataset: rock.datasets.nyu_depth_v2.NYUv2Detection,
             encoder: rock.ssd.encoder.Encoder,
             device: torch.device = torch.device("cuda"),
             gt_path: str = 'data/eval/gt_box.json',
             dt_path: str = 'data/eval/pred_box.json',
             max_output: int = 100,
             show_all_cats: bool = False) -> Tuple[float, float]:
    """ Evaluates the network's output using COCOeval (adapted for the NYUv2 dataset)

    |
    Prints out the mAP and related metrics for the given model on the given dataset

    Args:
        model: network
        dataset: dataset on which to run eval
        encoder: encoder use to encode / decode the network's output
        device: device on which to run eval (default: cuda)
        gt_path: save path for ground truths bounding boxes json file
        dt_path: save path for predicted bounding boxes json file
        max_output: maximum number of bounding boxes to consider per image (default: 100)
        show_all_cats: whether to show AP for all categories or just an average  (default: False)

    Returns:
        Average Precision (AP) @[ IoU=0.50:0.95] and AP @[ IoU=0.50]
    """
    # Put model in eval mode
    model.eval()

    _create_coco_files(model, dataset, encoder, device, gt_path, dt_path, max_output)

    cocoGt = COCO(gt_path)
    cocoDt = cocoGt.loadRes(dt_path)

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')

    if not show_all_cats:
        E.evaluate()
        E.accumulate()
        E.summarize()
        print("Current AP: {:.5f}".format(E.stats[0]))

        # Put model back in training mode
        model.train()

        return E.stats[0], E.stats[1]

    else:
        # Evaluation by category
        catIds = E.params.catIds
        cats = dataset.categories

        for elem in catIds:
            E.params.catIds = elem

            print("catId: " + str(elem))
            print("catName: " + cats[elem])
            E.evaluate()
            E.accumulate()
            E.summarize()
            print()

        print("All catIds: " + str(catIds))
        E.params.catIds = catIds
        E.evaluate()
        E.accumulate()
        E.summarize()
        print("Current AP: {:.5f}".format(E.stats[0]))

        # Put model back in training mode
        model.train()

        return E.stats[0], E.stats[1]  # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]


def _create_coco_files(model: torch.nn.Module,
                       dataset: rock.datasets.nyu_depth_v2.NYUv2Detection,
                       encoder: rock.ssd.encoder.Encoder,
                       device: torch.device,
                       gt_path: str,
                       dt_path: str,
                       max_output: int) -> None:
    # Create paths if they don't exist
    if not Path(gt_path).exists():
        Path(os.path.dirname(gt_path)).mkdir(parents=True, exist_ok=True)

    if not Path(dt_path).exists():
        Path(os.path.dirname(dt_path)).mkdir(parents=True, exist_ok=True)

    img_width, img_height = 640, 480

    gt_dict = _init_gt_dict(dataset)
    dt_dict = []

    model.to(device)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    for nbatch, sample in enumerate(dataloader, start=1):
        print("Parsing batch: {}/{}".format(nbatch, len(dataloader)), end='\r')
        with torch.no_grad():
            inp = sample['img'].to(device)
            img_id = sample['img_id']

            ploc, plabel, *aux_out = model(inp)

            for id in img_id.tolist():

                gt_dict['images'].append({"id": id, "width": img_width, "height": img_height, "file_name": None})

                boxes, labels = dataset.get_eval(id)
                for i, (box, label) in enumerate(zip(boxes, labels)):
                    annot = {}
                    annot['id'] = id * 100 + i
                    annot['image_id'] = id
                    annot['category_id'] = label
                    annot['bbox'] = [int(elem) for elem in list(box)]
                    annot['area'] = int(box[2]) * int(box[3])
                    annot['iscrowd'] = 0
                    annot['segmentation'] = None
                    gt_dict['annotations'].append(annot)

            dec = encoder.decode_batch(ploc, plabel, max_output_num=max_output)

            for id, preds in zip(img_id.tolist(), dec):
                pred_boxes, labels, confs = preds

                # If nothing predicted, don't add anything
                if pred_boxes.shape[0] == 0:
                    continue

                # Get box sizes in pixel and convert to l,t,w,h for COCOeval
                boxes = pred_boxes.cpu() * torch.tensor([img_width, img_height, img_width, img_height])
                l, t, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
                boxes[:, 0] = l
                boxes[:, 1] = t
                boxes[:, 2] = w
                boxes[:, 3] = h
                boxes = torch.round(boxes * 10) / 10

                for box, label, conf in zip(boxes.tolist(), labels.tolist(), confs.tolist()):
                    annot = {}
                    annot['image_id'] = id
                    annot['category_id'] = label
                    annot['bbox'] = box
                    annot['score'] = conf
                    dt_dict.append(annot)

    with open(gt_path, 'w') as outfile:
        json.dump(gt_dict, outfile)

    with open(dt_path, 'w') as outfile:
        json.dump(dt_dict, outfile)


# noinspection PyDictCreation
def _init_gt_dict(dataset: rock.datasets.nyu_depth_v2.NYUv2Detection) -> Dict[str, Union[List[Any], Dict[str, str]]]:
    d = {}
    d['info'] = {"description": "Subset of NYUv2 Dataset",
                 "url": "",
                 "version": "",
                 "year": 2020,
                 "contributor": "",
                 "date_created": ""}
    d['licenses'] = []
    d['images'] = []
    d['annotations'] = []
    d['categories'] = []

    # Remove background as a category
    for i, name in enumerate(dataset.categories[1:], start=1):
        d['categories'].append({"supercategory": "home", "id": int(i), "name": name})

    return d
