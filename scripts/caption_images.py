import torch
import json
import os
from PIL import Image


from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

filename = "datasets/redcaps/annotations_test/fashionadvice_2022-07_filtered_badwords_finalsummary_doc.json"

with open(filename, 'r') as f:
    data = json.load(f)
    threads = data['threads']

# image_paths = list()
images = list()
for thread in threads:
    image_path = f"datasets/redcaps/images/fashionadvice/{thread['submission_id']}.jpg"
    # image_paths.append(image_path)

    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
    images.append(i_image)

pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

output_ids = model.generate(pixel_values, **gen_kwargs)

preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
preds = [pred.strip() for pred in preds]

# count = 0
# for thread in threads:
#     thread['imgcap_vitgpt2'] = preds[count]
#     count += 1
#     print(preds[count])
new_filename = "caps_test.txt"
with open(new_filename, 'w') as f:
    for pred in preds:
        f.write("%s\n" % pred)
    print('done')