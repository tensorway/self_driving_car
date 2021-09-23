#%%
import numpy as np
import cv2
from machine_learning.models import RoadDetector
from machine_learning.utils import load_model
from PIL import Image
import random

class Base:
    '''
    simple base for agents
    provides visualization and model loading
    '''
    def __init__(self, device):
        self.model = RoadDetector()
        load_model(self.model, 'machine_learning/model_checkpoints/model_vit.pt')
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.arrow_right = np.array(Image.open('assets/arrow_right.png').convert('RGBA'))
        self.arrow_left  = np.array(Image.open('assets/arrow_left.png').convert('RGBA'))

    def predict_and_visualize(self, img):
        diff, overlaid = self.predict(img, self.device, visualize=True)
        xsize = min(int(diff*3/self.coeff), 224//2-1)
        ysize = 20
        midpoint = 224//2

        #TODO fuse both outcomes
        if xsize > 0:
            arrow = cv2.resize(self.arrow_right, (xsize, ysize))
            for i in range(0, ysize):
                for j in range(0, xsize):
                    if arrow[i, j, -1] != 0:
                        overlaid[i, j+midpoint, :] = np.array([255, 255, 0])
        else:
            arrow = cv2.resize(self.arrow_right, (-xsize, ysize))
            for i in range(0, ysize):
                for j in range(0, -xsize):
                    if arrow[i, j, -1] != 0:
                        overlaid[i, -j+midpoint, :] = arrow[i, j, :3] 

        return diff, overlaid

class RandomAgent(Base):
    '''
    random for degubbing
    '''
    def __init__(self, device):
        super().__init__(device)

    def predict(self, img, visualize=False):
        return (random.random()-0.5)*3.14/10
    
class LineFollower(Base):
    '''
    works like a basic line following
    robot. If the center of the line
    at a point is left from the center
    of the car it goes left
    if it is right it goes right
    if it is in the center it goes straight
    '''
    def __init__(self, device, coeff=1) -> None:
        super().__init__(device)
        self.coeff = coeff

    def predict(self, img, visualize=False):
        overlaid, _, line = self.model.visualize(img, self.device)
        diff = line[175:200].mean() - 224/2 
        if visualize:
            return diff*self.coeff, overlaid
        return diff*self.coeff

class CenterOfMassFollower(Base):
    '''
    tracks center of mass of predicted road
    If the center is left from the center
    of the car it goes left
    if it is right it goes right
    if it is in the center it goes straight
    '''
    def __init__(self, device, coeff=1) -> None:
        super().__init__(device)
        self.coeff = coeff

    def predict(self, img, visualize=False):
        overlaid, preds, line = self.model.visualize(img, self.device)
        pred = preds[0, :, :, 0].detach().cpu().numpy()
        pred += preds[0, :, :, 1].detach().cpu().numpy()
        weights = np.sum(pred, axis=1)
        diff = (line@weights)/np.sum(weights) - 224/2 
        if visualize:
            return diff*self.coeff, overlaid
        return diff*self.coeff

# %%
