import argparse
import glob

import h5py
import numpy as np
import os

import torch
from PIL import Image as PIL_Image
from h5py import string_dtype
from torch import nn

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm

from utils import RESNET_IMG_FEATS_DIM, H5_IDS_KEY, DATA_DIR_IMAGENET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str)
    args = parser.parse_args()

    for split in ["train", "val"]:
        base_path = os.path.join(args.source_dir, split)

        with h5py.File(os.path.join(DATA_DIR_IMAGENET, f"{split}_features.hdf5"), 'w') as h5_db:
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            modules = list(resnet.children())[:-1]
            model = nn.Sequential(*modules)
            for p in model.parameters():
                p.requires_grad = False
            model = model.to(device)

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            preprocessing = transforms.Compose([
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

            all_ids = []
            ids_batch = []
            images_batch = []

            for filepath in tqdm(glob.iglob(f'{base_path}/**/*.JPEG')):
                img_id = os.path.basename(filepath).replace(".JPEG", "")

                img = PIL_Image.open(filepath)
                images_batch.append(img)
                ids_batch.append(img_id)
                all_ids.append(img_id)

                if len(images_batch) == BATCH_SIZE:
                    images = [preprocessing(img) for img in images_batch]

                    images = torch.stack(images).to(device)

                    feats = model(images).squeeze().cpu().numpy()

                    for id, feat in zip(ids_batch, feats):
                        h5_features = h5_db.create_dataset(id, RESNET_IMG_FEATS_DIM, dtype=np.float32)

                        h5_features[:] = feat

                    images_batch = []
                    ids_batch = []

            h5_ids = h5_db.create_dataset(H5_IDS_KEY, len(all_ids), dtype=string_dtype())
            h5_ids[:] = all_ids
