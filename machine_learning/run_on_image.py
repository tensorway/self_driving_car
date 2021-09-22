#%%
import os
import glob
import torch
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch as th
from pathlib import Path
from models import RoadDetector
from dataset import SelfDrivingDataset
from transforms import img_transfrom_train, label_transfrom_train
from utils import load_model
import logging

MODEL_PATH ='model_checkpoints/model_vit.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

# %%
model = RoadDetector()

load_model(model, str(MODEL_PATH))
model.to(device);

#%%

if 0:
    dataset = SelfDrivingDataset(
        dataset_path="dataset", 
        img_transform=img_transfrom_train, 
        label_transform=label_transfrom_train, 
        split='train'
        )
    idx = random.randint(0, len(dataset))
    print("dataset at", idx)
    img = dataset[idx][0]

else:
    test_images = glob.glob('realworld_eval_data/*.png')
    # test_images = glob.glob('../self_driving_tractor/simdata/IMG/center*')
    test_images.sort(key=os.path.getmtime)
    toopen = random.choice(test_images) #"/home/darijan/Pictures/dubravakoso.png"
    img = Image.open(toopen).convert('RGB')

with th.no_grad():
    overlaid, preds, line = model.visualize(img, device)


plt.imshow(overlaid);
save_name = 'realworld_outputs/'+toopen.split('/')[-1]
plt.imsave(save_name, overlaid)

