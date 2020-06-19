import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import rock.ssd.prior_boxes
import rock.ssd.encoder
import rock.datasets.image_folder
import rock.model.network


def add_graph_to_tb(writer_path: str = 'data/runs/rock_model'):

    print("Adding graph to Tensorboard")
    model = rock.model.network.rock_network()
    dataset = rock.datasets.image_folder.ImageFolder('data/detection/images')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    images, _ = next(iter(dataloader))

    model = model.cuda()
    images = images.cuda()

    writer = SummaryWriter(writer_path)
    writer.add_graph(model, images)
    writer.close()
    print("Done adding graph!")
