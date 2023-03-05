import argparse

import h5py
import numpy as np
import os

import torch
from fiftyone.zoo import load_zoo_dataset
from PIL import Image as PIL_Image
import matplotlib.pyplot as plt
from h5py import string_dtype
from matplotlib.patches import Rectangle
from torch import nn

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from tqdm import tqdm

from utils import GUESSWHAT_IMG_FEATS_DIM, GUESSWHAT_H5_IDS_KEY, GUESSWHAT_MAX_NUM_OBJECTS, DATA_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_image(
    img_data,
    detections=None,
):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.imshow(img_data)

    ax = plt.gca()
    if detections:
        for detection in detections:
            bb = detection.bounding_box
            ax.add_patch(
                Rectangle(
                    (bb[0] * img_data.width, bb[1] * img_data.height),
                    bb[2] * img_data.width,
                    bb[3] * img_data.height,
                    fill=False,
                    edgecolor="red",
                    linewidth=3,
                )
            )

    plt.tick_params(labelbottom="off", labelleft="off")
    plt.show()


def get_spatial_feat(bbox, im_width, im_height):
    # TODO: use?
    # Rescaling
    x_left = (1.*bbox.x_left / im_width) * 2 - 1
    x_right = (1.*bbox.x_right / im_width) * 2 - 1
    x_center = (1.*bbox.x_center / im_width) * 2 - 1

    y_lower = (1.*bbox.y_lower / im_height) * 2 - 1
    y_upper = (1.*bbox.y_upper / im_height) * 2 - 1
    y_center = (1.*bbox.y_center / im_height) * 2 - 1

    x_width = (1.*bbox.x_width / im_width) * 2
    y_height = (1.*bbox.y_height / im_height) * 2

    feat = [x_left, y_lower, x_right, y_upper, x_center, y_center, x_width, y_height]
    feat = np.array(feat)

    return feat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-area-in-pixels", type=int, default="500")
    args = parser.parse_args()

    for split in ["train", "validation"]:
        ds = load_zoo_dataset("coco-2014", split=split)

        with h5py.File(os.path.join(DATA_DIR, split + "_features.hdf5"), 'w') as h5_db:
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            modules = list(resnet.children())[:-1]
            model = nn.Sequential(*modules)
            for p in model.parameters():
                p.requires_grad = False
            model = model.to(device)

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            preprocessing = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])

            ids = []
            for sample in tqdm(ds):
                img = PIL_Image.open(sample.filepath)
                if not sample.ground_truth or img.mode != 'RGB':
                    continue

                cropped_objects = []
                for detection in sample.ground_truth.detections:
                    bb = detection.bounding_box

                    # crop image
                    cropped = img.crop(
                        (
                            bb[0] * img.width,
                            bb[1] * img.height,
                            bb[2] * img.width + bb[0] * img.width,
                            bb[3] * img.height + bb[1] * img.height,
                        )
                    )

                    area_ppx = (bb[2] * img.width) * (bb[3] * img.height)
                    if area_ppx > args.min_area_in_pixels:
                        cropped_objects.append(cropped)

                if len(cropped_objects) < 2:
                    # Do not consider images with only one object
                    continue

                if len(cropped_objects) > GUESSWHAT_MAX_NUM_OBJECTS:
                    cropped_objects = cropped_objects[:GUESSWHAT_MAX_NUM_OBJECTS]

                # The first image in the tensor is the overview, all following are the cropped objects
                images = [img] + cropped_objects
                images = [preprocessing(img) for img in images]

                images = torch.stack(images).to(device)

                feats = model(images).squeeze().cpu().numpy()

                h5_features = h5_db.create_dataset(sample.id, (len(images), GUESSWHAT_IMG_FEATS_DIM), dtype=np.float32)

                h5_features[:] = feats

                ids.append(sample.id)

            h5_ids = h5_db.create_dataset(GUESSWHAT_H5_IDS_KEY, len(ids), dtype=string_dtype())
            h5_ids[:] = ids
