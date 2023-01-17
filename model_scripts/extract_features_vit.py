import os
import sys
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import argparse

from transformers import ViTImageProcessor, ViTModel

parser = argparse.ArgumentParser()
default_imgs = "/gallery_tate/keighley.overbay/thread-summarization/thread_data/full_dataset/images"
parser.add_argument("--model", type=str, default="ViT-B/16")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--img_dir", type=str, default=default_imgs)
parser.add_argument("--out_dir", type=str, default="./images_features")

args = parser.parse_args()

device = args.device
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

images = sorted(
   [os.path.join(args.img_dir, image) for image in list(os.listdir(args.img_dir))]
)

print(len(images))

for image in tqdm(images):
    image_obj = Image.open(image)
    # images_obj = preprocess(image_obj).unsqueeze(0).to(device)

    image_features = feature_extractor(image_obj, return_tensors="pt")

    out_file = os.path.join(
        args.out_dir, os.path.basename(image)[:-4]
    )
    np.save(out_file, image_features)