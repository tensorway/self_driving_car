#%%
from torch.utils.data import Dataset
import glob
from PIL import Image
from pathlib import Path


class SelfDrivingDataset(Dataset):
    def __init__(self, dataset_path, split='train', img_transform=None, label_transform=None):
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.examples = []
        dataset_path = Path(dataset_path)
        img_files = glob.glob(str(dataset_path/'bdd100k/images/100k'/split/'*'))
        img_files.sort()
        for img_file in img_files:
            name = img_file.split('/')[-1].split('.')[0] 
            label_file = str(dataset_path/'bdd100k/labels/drivable/masks'/split/(name+'.png'))  # images are in jpg masks in png
            self.examples.append((img_file, label_file))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        img = Image.open(self.examples[idx][0])
        label = Image.open(self.examples[idx][1])
        if self.img_transform:
            img = self.img_transform(img)
        if self.label_transform:
            label = self.label_transform(label)
        return img, label

#%%
