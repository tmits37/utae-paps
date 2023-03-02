import sys
sys.path.append('/nas/k8s/dev/research/doyoungi/croptype_cls/utae-paps')

import argparse
import json
import os
import pprint
import rasterio
import gdal

from glob import glob
from tqdm import tqdm 
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt

from src import utils, model_utils
from src.dataset import PASTIS_Dataset, HaenamDataset

from train_semantic import iterate, overall_performance, save_results, prepare_output


parser = argparse.ArgumentParser()
parser.add_argument(
    "--weight-folder",
    type=str,
    default="",
    help="Path to the main folder containing the pre-trained weights",
)
parser.set_defaults(cache=False)

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def get_seed(raster, length, overlap):
    diff = length - overlap

    X_seed = np.arange(raster.height // diff + 1) * diff
    X_seed[-1] = raster.height - length

    Y_seed = np.arange(raster.width // diff + 1) * diff
    Y_seed[-1] = raster.width - length
    if abs(X_seed[-1] - X_seed[-2]) < int(length / 2):
        X_seed = np.delete(X_seed, [-2])
    if abs(Y_seed[-1] - Y_seed[-2]) < int(length / 2):
        Y_seed = np.delete(Y_seed, [-2])

    X_seed, Y_seed = np.meshgrid(X_seed, Y_seed)
    positions = np.vstack([X_seed.ravel(), Y_seed.ravel()])

    seed = []
    for x, y in zip(X_seed.ravel(), Y_seed.ravel()):
        seed.append(np.array([x, y]))
    return seed

class TestHaenamDataset(torch.utils.data.Dataset):
    def __init__(self,
                 folder="/nas/k8s/dev/research/doyoungi/dataset/ads_22a/normed"
                 ):
        super(TestHaenamDataset, self).__init__()
        tif_list = glob(folder + "/*.tif")
        rasters = rasterio.open(tif_list[0])
        self.seed = get_seed(rasters, length=32, overlap=0)

        tif_list = sorted(tif_list)
        self.src_list = [gdal.Open(x) for x in tif_list]

        self.mean=[103.53, 116.28, 123.675, 123.675],
        self.std=[57.375, 57.12, 58.395, 58.395],
        
    def __len__(self):
        return len(self.seed)

    def __getitem__(self, item):
        y, x = self.seed[item]
        
        img_list = [src.ReadAsArray(int(x), int(y), 32, 32) for src in self.src_list]
        resized_img = [cv2.resize(img.transpose(1,2,0), dsize=(0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR) for img in img_list]
        data = np.stack(resized_img, axis=0).transpose(0,3,1,2).astype('float32')

        # norm data
        mean = np.array(self.mean)[..., None, None] # [np.newaxis, :]
        std = np.array(self.std)[..., None, None] # [np.newaxis, :]

        data = (data - mean) / std

        data = torch.from_numpy(data).float() # .type(torch.FloatTensor)
        dates = torch.tensor([296, 301, 306, 311, 326, 331, 336]) # fixed to 7
        # target = torch.from_numpy(target)
        # float32, int64, uint8

        return (data, dates), (x, y)


if __name__ == '__main__':
    args = parser.parse_args()

    weight_folder = args.weight_folder
    save_folder = os.path.join(weight_folder, 'scene_infernece')
    os.makedirs(save_folder, exist_ok=True)

    with open(os.path.join(weight_folder, "conf.json")) as file:
        model_config = json.loads(file.read())
    
    config = {**model_config, 'dataset_folder': '/nas/k8s/dev/research/doyoungi/dataset/ads_22a/pastis_style'}
    config = argparse.Namespace(**config)
    config.fold = None
    device = 'cuda'

    model = model_utils.get_model(config, mode="semantic")
    model = model.to(device)

    sd = torch.load(
        os.path.join(weight_folder, "Fold_{}".format(1), "model.pth.tar"),
        map_location=device,
    )
        
    model.load_state_dict(sd["state_dict"])
    print('model loaded')

    # Loss
    weights = torch.Tensor([1, 2]).to('cuda').float()
    criterion = nn.CrossEntropyLoss(
                weight=weights,
                ignore_index=255
                )
    dt_test = TestHaenamDataset()
    collate_fn = lambda x: utils.pad_collate(x, pad_value=0)
    test_loader = data.DataLoader(
        dt_test,
        batch_size=24,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model.eval() 

    for i, batch in enumerate(tqdm(test_loader)):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), (x_tic, y_tic) = batch

        with torch.no_grad():
            out = model(x, batch_positions=dates)
            
        with torch.no_grad():
            pred = out.argmax(dim=1)
            
        batch_size = x.shape[0]
        for b in range(batch_size):
            col = x_tic[b].item()
            row = y_tic[b].item()
            prediction = pred[b].detach().cpu().numpy()

            prediction = prediction.astype('uint8')
            # low_prediction = cv2.resize(prediction, dsize=(32,32), interpolation=cv2.INTER_LINEAR)
            basename = f'{col}_{row}.npy'
            np.save(os.path.join(save_folder, basename), prediction)


    print('Done!')