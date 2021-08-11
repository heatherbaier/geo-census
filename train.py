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

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from utils import *



if __name__ == "__main__":
    
    json_dir = "../data/infra_vars/"

    for json_file in os.listdir(json_dir)[16:]:
            
        var_name = json_file.split(".")[0]
        full_path = os.path.join(json_dir, json_file)
        
        for kfold in range(0, 5):
            
            test = var_name + "_" + str(kfold) + "_" + "stats.txt"
            
            if os.path.exists(test):
                
                print("Skipping because already done!")
                
                continue   
                
            print("\n")        
        
            # Prep data
            with open(full_path) as m:
                mig_data = json.load(m)
            mig_data = pd.DataFrame.from_dict(mig_data, orient = 'index').reset_index()
            mig_data.columns = ['muni_id', 'var']

            image_names = get_png_names("../../mex_imagery/")
            y = get_y(image_names, mig_data)
            
            open("stats.txt", "w").close()

            train_num = int(len(image_names) * .75)
            train_indices = random.sample(range(len(image_names)), train_num)
            val_indices = [i for i in range(len(image_names)) if i not in train_indices]
            train_names = [image_names[i] for i in train_indices]
            val_names = [image_names[i] for i in val_indices]
            
            val_file_name = var_name + "_" + str(kfold) + "_" + "valimages.txt"
            validation_images = [image_names[i] for i in val_indices]
            with open(val_file_name, 'w') as f:
                for item in validation_images:
                    f.write("%s\n" % item)            

            batch_size = 1
            train = [(torchvision.transforms.functional.adjust_brightness(load_inputs(image_names[i]), brightness_factor = 2).squeeze(), y[i]) for i in train_indices]
            val = [(torchvision.transforms.functional.adjust_brightness(load_inputs(image_names[i]), brightness_factor = 2).squeeze(), y[i]) for i in val_indices]
            train_dl = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
            val_dl = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = True)        

            print("Num training: ", len(train_dl))
            print("Num validation: ", len(val_dl))

            os.mkdir("ckpt")
            
            config, unparsed = get_config()
            trainer = Trainer(config, (train_dl, val_dl))
            trainer.train()
            
            ckpt_renamed = var_name + "_" + str(kfold) + "_" + "trained"
            os.rename("ckpt", ckpt_renamed) 
            
            stats_renamed = var_name + "_" + str(kfold) + "_" + "stats.txt"
            os.rename("stats.txt", stats_renamed)     
            
            del config
            del unparsed
            del trainer

            
            
            