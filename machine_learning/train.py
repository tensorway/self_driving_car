#%%
import time
import torch
import torch as th
from pathlib import Path
from clearml import Task, Logger
import torch.nn.functional as F
import torchvision
from models import RoadDetector
from dataset import SelfDrivingDataset
from torch.utils.data import DataLoader
from transforms import img_transfrom_train, label_transfrom_train
from utils import load_model, save_model, collate, overlay_predictions_on_images

MODEL_CHECKPOINTS_PATH = Path('/home/darijan/Downloads')
MODEL_NAME = 'vit'
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
OPTIMIZER_PATH = MODEL_CHECKPOINTS_PATH/('optimizer_'+MODEL_NAME+'.pt')
SAVE_DELTA = 20*60 #in seconds

# task = Task.init(project_name="Self driving car", task_name="test")
# logger = Logger.current_logger()

#%%
train_dataset = SelfDrivingDataset(
    dataset_path="dataset", 
    img_transform=img_transfrom_train, 
    label_transform=label_transfrom_train, 
    split='train'
    )
valid_dataset = SelfDrivingDataset(
    dataset_path="dataset", 
    img_transform=img_transfrom_train, 
    label_transform=label_transfrom_train, 
    split='val'
    )
visualization_imgs = [valid_dataset[i] for i in range(6)]

train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=1, 
                    shuffle=True, 
                    num_workers=0, 
                    collate_fn=collate,
                    pin_memory=False, 
)
valid_dataloader = DataLoader(
                    valid_dataset, 
                    batch_size=1, 
                    shuffle=False, 
                    num_workers=0, 
                    collate_fn=collate,
                    pin_memory=False, 
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

# %%
model = RoadDetector()
load_model(model, str(MODEL_PATH))
model.to(device);
#%%%
opt = th.optim.Adam([
    {'params':model.parameters(), 'lr':1e-4},
])
load_model(opt, str(OPTIMIZER_PATH))
# %%
step = 0
t_last_save = time.time()
for ep in range(10):
    for ibatch, (imgs, labels) in enumerate(train_dataloader):
        preds = model(imgs, device)

        labels = labels.to(device)
        labels_hot = F.one_hot(labels, preds.shape[-1])
        loss = - (labels_hot * th.log(preds+1e-8)).mean()
        acc = (labels == th.argmax(preds, dim=-1)).float().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        logger.report_scalar("loss", "train", iteration=step , value=loss.item())
        logger.report_scalar("acc ", "train", iteration=step , value=acc.item())

        if ibatch%10 == 0:
            print(ep, ibatch, loss.item())
        if ibatch % 10 == 0:
            for ibatch, (imgs, labels) in enumerate(valid_dataloader):
                with th.no_grad():
                    preds = model(imgs, device)
                    labels = labels.to(device)
                    labels_hot = F.one_hot(labels, preds.shape[-1])
                    loss = - (labels_hot * th.log(preds+1e-8)).mean()
                    acc = (labels == th.argmax(preds, dim=-1)).float().mean()
                    logger.report_scalar("loss", "valid", iteration=step , value=loss.item())
                    logger.report_scalar("acc ", "valid", iteration=step , value=acc.item())
                break
            if ibatch % 100:
                with th.no_grad():
                    preds = model(visualization_imgs, device)
                overlaid = overlay_predictions_on_images(visualization_imgs, preds)
                for i, img in enumerate(overlaid):
                    logger.report_image("imgs", str(i), iteration=step, image=img)

        if time.time() - t_last_save > SAVE_DELTA:
            save_model(model, str(MODEL_PATH))
            save_model(opt, str(OPTIMIZER_PATH))

        step += 1
