import argparse

import h5py
import numpy as np
import os

import torch
from fiftyone.zoo import load_zoo_dataset
from PIL import Image as PIL_Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch import nn

from torchvision.models import resnet50, ResNet50_Weights, ViT_B_16_Weights
from torchvision.models.vision_transformer import vit_b_16
from tqdm import tqdm

from utils import RESNET_IMG_FEATS_DIM, ViT_IMG_FEATS_DIM, GUESSWHAT_MAX_NUM_OBJECTS, DATA_DIR_GUESSWHAT

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
    parser.add_argument("--model", type=str, default="vit")

    args = parser.parse_args()

    for split in ["train", "validation"]:
        ds = load_zoo_dataset("coco-2014", split=split)

        with h5py.File(os.path.join(DATA_DIR_GUESSWHAT, f"{split}_features_{str(args.min_area_in_pixels)}_{args.model}.hdf5"), 'w') as h5_db:
            if args.model == "resnet":
                resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
                modules = list(resnet.children())[:-1]
                model = nn.Sequential(*modules)

                preprocessing = ResNet50_Weights.DEFAULT.transforms()
                feat_size = RESNET_IMG_FEATS_DIM
            elif args.model == "vit":
                model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

                preprocessing = ViT_B_16_Weights.DEFAULT.transforms()
                feat_size = ViT_IMG_FEATS_DIM
            else:
                raise RuntimeError("Unknown model: ", args.model)

            for p in model.parameters():
                p.requires_grad = False
            model = model.to(device)

            ids = []
            print("\nExtracting features: ")
            for sample in tqdm(ds):
                if not sample.ground_truth:
                    continue

                width = sample.metadata.width
                height = sample.metadata.height

                bbs = []
                for detection in sample.ground_truth.detections:
                    bb = detection.bounding_box

                    area_ppx = (bb[2] * width) * (bb[3] * height)
                    if area_ppx > args.min_area_in_pixels:
                        bbs.append(bb)

                if len(bbs) < 2:
                    # Do not consider images with only one object
                    continue

                if len(bbs) > GUESSWHAT_MAX_NUM_OBJECTS:
                    bbs = bbs[:GUESSWHAT_MAX_NUM_OBJECTS]

                img = PIL_Image.open(sample.filepath)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                cropped_objects = []
                for bb in bbs:
                    # crop image
                    cropped = img.crop(
                        (
                            bb[0] * img.width,
                            bb[1] * img.height,
                            bb[2] * img.width + bb[0] * img.width,
                            bb[3] * img.height + bb[1] * img.height,
                        )
                    )
                    cropped_objects.append(cropped)

                # The first image in the tensor is the overview, all following are the cropped objects
                images = [img] + cropped_objects
                images = [preprocessing(img) for img in images]

                images = torch.stack(images).to(device)

                if args.model == "resnet":
                    feats = model(images).squeeze().cpu().numpy()
                else:
                    feats = model._process_input(images)

                    # Expand the class token to the full batch
                    batch_class_token = model.class_token.expand(images.shape[0], -1, -1)
                    feats = torch.cat([batch_class_token, feats], dim=1)

                    feats = model.encoder(feats)

                    # We're only interested in the representation of the classifier token that we appended at pos 0
                    feats = feats[:, 0].squeeze().cpu().numpy()

                h5_features = h5_db.create_dataset(sample.id, (len(images), feat_size), dtype=np.float32)

                h5_features[:] = feats

