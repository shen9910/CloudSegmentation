import yaml
import torch
import random
import os
import numpy as np
import pandas as pd
import cv2
from collections import OrderedDict


BANDS = ["B02", "B03", "B04", "B08"]

def jaccard_metric(pred, actual):
    # get the sum of intersection and union over all chips
    intersection = 0
    union = 0

    intersection += np.logical_and(actual, pred).sum()
    union += np.logical_or(actual, pred).sum()

    # calculate the score across all chips
    iou = intersection / union 
    return iou

def makeTrainCSV(train_path):
    train_features = os.path.join(train_path, "train_features")
    train_labels = os.path.join(train_path, "train_labels")
    cases = os.listdir(train_features)
    data = []
    for case in cases:
        path = os.path.join(train_labels, case+".tif")
        if not os.path.exists(path):
            print(f'Label for {case} chip_id does not exist')
            continue
        label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        percentage = round(label.sum()/(512*512) ,1)
        data.append({
            'chip_id': case,
            f'{BANDS[0]}_path': os.path.join(train_features, case, BANDS[0]+".tif"),
            f'{BANDS[1]}_path': os.path.join(train_features, case, BANDS[1]+".tif"),
            f'{BANDS[2]}_path': os.path.join(train_features, case, BANDS[2]+".tif"),
            "label": os.path.join(train_labels, case + ".tif"),
            "segmentedArea": percentage
        })

    print("Total number of training cases: ", len(data))
    return pd.DataFrame(data)
    

def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_pytorch_model(state_dict, *args, **kwargs):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('model.'):
            name = name.replace('model.', '') # remove `model.`
        new_state_dict[name] = v
    return new_state_dict


def convert_model(ckpt_path, root_dir):
    ckpt_path = str(ckpt_path)
    version_name, filename = '', os.path.basename(ckpt_path)[:-5]
    weights_name = os.path.join(root_dir, f"{filename.split('.')[0]}{version_name}.pt")
    ckpt_dict = torch.load(ckpt_path)
    best_model = load_pytorch_model(ckpt_dict['state_dict'])
    torch.save(best_model, weights_name)