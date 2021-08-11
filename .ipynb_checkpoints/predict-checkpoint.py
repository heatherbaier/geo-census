from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value
from model import RecurrentAttention
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import AverageMeter
from config import get_config
from utils import plot_images
from trainer import Trainer
import sklearn.metrics
from tqdm import tqdm
import pandas as pd
import torchvision
import numpy as np
import argparse
import random
import pickle
import shutil
import torch
import time



from utils import *






if __name__ == "__main__":
    
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("variable", help = "Census variable")
    parser.add_argument("gpu", help = "GPU")
    args = parser.parse_args()
    
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    for kfold in range(0, 5):

        print(args.variable + " kfold: ", kfold)

        try:
            m = open("../data/socio_vars/" + args.variable + ".json",)
        except:
            m = open("../data/infra_vars/" + args.variable + ".json",)
            
        mig_data = json.load(m)
        m.close()
        mig_data = pd.DataFrame.from_dict(mig_data, orient = 'index').reset_index()
        mig_data.columns = ['muni_id', 'var']
        mig_data.head()

        image_names = get_png_names("../../mex_imagery/")

        with open("./final/val_images/" + args.variable + "_" + str(kfold) + "_valimages.txt") as ims:
            val_names = ims.read().splitlines()

        train_names = [i for i in image_names if i not in val_names]

        y = get_y(image_names, mig_data)

        train_num = int(15 * .70)
        train_indices = random.sample(range(0, 15), train_num)
        val_indices = [i for i in range(0, 15) if i not in train_indices]

        batch_size = 1
        train = [(torchvision.transforms.functional.adjust_brightness(load_inputs(image_names[i]), brightness_factor = 2).squeeze(), y[i]) for i in train_indices]
        val = [(torchvision.transforms.functional.adjust_brightness(load_inputs(image_names[i]), brightness_factor = 2).squeeze(), y[i]) for i in val_indices]
        train_dl = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
        val_dl = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = True)


        config, unparsed = get_config()
        trainer = Trainer(config, (train_dl, val_dl))

        checkpoint = torch.load("./final/trained_models/" + args.variable + "_" + str(kfold) + "_trained/ram_4_50x50_0.75_model_best.pth.tar")
        checkpoint = checkpoint["model_state"]


        trues, preds, tv = [], [], []
        it = 0

        for im in val_names:

            try:

                i = load_inputs(im)
                o2 = torch.tensor(mig_data[mig_data["muni_id"] == str(im.split("/")[3].split("_")[0].split(".")[0])]['var'].values[0])

                pred = trainer.predict(1, i, o2, checkpoint).item()
                true = o2.item()

                trues.append(true)
                preds.append(pred)
                tv.append('val')

                it += 1
                print(it, end = "\r")

            except Exception as e:

                print(e, im)

        for im in train_names:

            try:

                i = load_inputs(im)
                o2 = torch.tensor(mig_data[mig_data["muni_id"] == str(im.split("/")[3].split("_")[0].split(".")[0])]['var'].values[0])

                pred = trainer.predict(1, i, o2, checkpoint).item()
                true = o2.item()

                trues.append(true)
                preds.append(pred)
                tv.append('train')  

                it += 1
                print(it, end = "\r")

            except Exception as e:

                print(e, im)




        preds_df = pd.DataFrame()
        preds_df['true'], preds_df['pred'] = trues, preds
        preds_df['tv'] = tv
        preds_df["error"] = abs(preds_df["true"] - preds_df["pred"])

        def mape(actual, pred): 
            actual, pred = np.array(actual), np.array(pred)
            return np.mean(np.abs((actual - pred) / actual)) * 100

        def mpe(actual, pred): 
            actual, pred = np.array(actual), np.array(pred)
            return np.mean((actual - pred) / actual) * 100

        print("R2: ", sklearn.metrics.r2_score(preds_df['true'], preds_df['pred']))
        print("MAE: ", sklearn.metrics.mean_absolute_error(preds_df['true'], preds_df['pred']))
        print("MAPE: ", mape(preds_df['true'], preds_df['pred']))
        print("MPE: ", mpe(preds_df['true'], preds_df['pred']))

        preds_df.to_csv("./final/preds/" + args.variable + "_" + str(kfold) + "_preds.csv")
