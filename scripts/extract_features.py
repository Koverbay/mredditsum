import numpy as np
import torch
import argparse
import glob
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models import resnext101_32x8d
from torchvision.models import vit_l_16
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTModel, CLIPProcessor
import pdb


def main(args):

    # Get feature extractor model
    # model = resnet50()
    if args.model == "resnext":
        model = resnext101_32x8d()
        modules = list(model.children())[:-1]
        feature_extractor = nn.Sequential(*modules).to('cuda')
        feature_extractor.out_channels = 2048

    # TODO: Finish implementing VIT embeddings + run tests
    if args.model == "vit":
        # model = vit_l_16(weights='DEFAULT')
        # modules = list(model.children())[:-1]
        # feature_extractor = nn.Sequential(*modules)
        # feature_extractor.out_channels = 2048

        # image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k").to('cuda')


    # Get list of image id's
    img_paths = glob.glob(f"{args.imgfolder}/*")
    for img_path in tqdm(img_paths):
        img_id = img_path.split('/')[-1][:-4]
        # Run it on image
        input_image = Image.open(img_path).convert('RGB')
        if args.model == "resnext":
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image).to('cuda')
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            with torch.no_grad():
                output = feature_extractor(input_batch)

            # Save to file
            filepath = f'../data/image_features/resnext101/{img_id}.npy'
            # filepath = f'/gallery_getty/jaewoo.ahn/multimodal-thread-sum/data/image_features/resnext101/{img_id}.npy'
            np.save(filepath, output.to('cpu').squeeze().unsqueeze(0))

        if args.model == "vit":
            inputs = image_processor(images=input_image, return_tensors="pt")['pixel_values'].to('cuda')
            with torch.no_grad():
                outputs = model(pixel_values=inputs)
            # output = outputs.last_hidden_state #
            output = outputs[1]
            # pdb.set_trace()

            # Save to file
            filepath = f'../data/image_features/vitb-32/{img_id}.npy'
            # filepath = f'/gallery_getty/jaewoo.ahn/multimodal-thread-sum/data/image_features/vitb-32/{img_id}.npy'
            np.save(filepath, output.to('cpu'))

        filepath = f'/gallery_getty/jaewoo.ahn/multimodal-thread-sum/data/image_features/vitb-32/{img_id}.npy'
        np.save(filepath, output.to('cpu')) #.squeeze().unsqueeze(0))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--imgfolder', required=True, type=str)
    parser.add_argument('-m', '--model', default="resnext", choices=['resnext','vit'])
    args = parser.parse_args()
    main(args)
