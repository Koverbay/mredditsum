
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

data_dict = json.load(open("thread_caption_comments.json"))
# print(data_dict)
img_f = '/gallery_tate/keighley.overbay/thread-summarization/thread_data/full_dataset/images'

img_list = [f for f in os.listdir(img_f)]
valid_ids = set()

id2steps = {}

for img in img_list:
    img_ids = img.split(".")[0]
    valid_ids.add(img_ids)
    id2steps[img_ids] = img

valid_ids = list(valid_ids)

X_train, X_test = train_test_split(valid_ids, test_size=300, random_state=42)
X_valid, X_test = train_test_split(X_test, test_size=0.5, random_state=42)

from datasets import Dataset, DatasetDict

ids = {"train": X_train, "valid": X_valid, "test": X_test}

dataset_dict = DatasetDict()
ddict = {}
for k in valid_ids:
    ddict = {"image":[],"caption_comments":[], "summary":[] }
    if k in data_dict.keys():

        ddict["image"] = id2steps[k]
        ddict["caption_comments"].append(data_dict[k]["cap_com"])
        ddict["summary"].append(data_dict[k]["edited_summaries"])
        

    dataset_dict[k] = Dataset.from_dict(ddict)


dataset_dict.save_to_disk("thread_sum_input_data")