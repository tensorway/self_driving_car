#%%
import os
import time
import glob
import torch
import random
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch as th
from pathlib import Path
import torchvision  
from models import RoadDetector
from dataset import SelfDrivingDataset
from transforms import img_transfrom_train, label_transfrom_train
from utils import load_model, save_model, collate, overlay_predictions_on_images

MODEL_CHECKPOINTS_PATH = Path('/home/darijan/Downloads')
MODEL_NAME = 'vit'
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
SAVE_DELTA = 20*60 #in seconds

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

dataset = SelfDrivingDataset(
    dataset_path="dataset", 
    img_transform=img_transfrom_train, 
    label_transform=label_transfrom_train, 
    split='train'
    )
# %%
model = RoadDetector()
#%%
load_model(model, str(MODEL_PATH))
model.to(device);
#%%
import cv2
test_images = glob.glob('realworld_eval_data/*.png')
# test_images = glob.glob('../self_driving_tractor/simdata/IMG/center*')
arrow_right = np.array(Image.open('assets/arrow_right.png').convert('RGBA'))
arrow_left  = np.array(Image.open('assets/arrow_left.png').convert('RGBA'))
test_images.sort(key=os.path.getmtime)
if 0:
    idx = random.randint(0, len(dataset))
    print("dataset at", idx)
    img = dataset[idx][0]
else:
    toopen = random.choice(test_images) #"/home/darijan/Pictures/dubravakoso.png"
    img = Image.open(toopen).convert('RGB')
with th.no_grad():
    overlaid, preds, line = model.visualize(img, device)


diff = line[150:175].mean() - 224/2 
xsize = min(int(diff*3), 224//2-1)
ysize = 20
midpoint = 224//2
if xsize > 0:
    arrow = cv2.resize(arrow_right, (xsize, ysize))
    for i in range(0, ysize):
        for j in range(0, xsize):
            if arrow[i, j, -1] != 0:
                overlaid[i, j+midpoint, :] = np.array([255, 255, 0])
else:
    arrow = cv2.resize(arrow_right, (-xsize, ysize))
    for i in range(0, ysize):
        for j in range(0, -xsize):
            if arrow[i, j, -1] != 0:
                overlaid[i, -j+midpoint, :] = arrow[i, j, :3] 

plt.imshow(overlaid);

# %%
import cv2
import numpy as np
import time
import math

name = 'car.mp4'
cap = cv2.VideoCapture('realworld_eval_data/'+name)
out = cv2.VideoWriter('realworld_outputs/'+name,cv2.VideoWriter_fourcc('M','P','4','V'), 60, (224,224))
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

while(cap.isOpened()):
    t = time.time()
    ret, frame = cap.read()
    if ret == True:
        overlaid, preds = model.visualize(frame, device, middles=False)

        # diff = line[150:175].mean() - 224/2 
        # xsize = min(int(diff*3), 224//2-1)
        # xsize = 1 if xsize==0 else xsize
        # ysize = 50
        # midpoint = 224//2
        # if xsize > 0:
        #     arrow = cv2.resize(arrow_right, (xsize, ysize))
        #     for i in range(0, ysize):
        #         for j in range(0, xsize):
        #             if arrow[i, j, -1] != 0:
        #                 overlaid[i, j+midpoint, :] = np.array([255, 0, 0])
        # else:
        #     arrow = cv2.resize(arrow_right, (abs(xsize), ysize))
        #     for i in range(0, ysize):
        #         for j in range(0, -xsize):
        #             if arrow[i, j, -1] != 0:
        #                 overlaid[i, -j+midpoint, :] = np.array([255, 0, 0])
        final = overlaid
        out.write(final)
        cv2.imshow('Frame', final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: 
        break
    print(1/(time.time()-t))

cap.release()
cv2.destroyAllWindows()
out.release()

# %%
