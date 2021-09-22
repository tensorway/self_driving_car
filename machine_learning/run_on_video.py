#%%
import cv2
import time
import torch
from models import RoadDetector
from utils import load_model

NAME = 'car.mp4'
cap = cv2.VideoCapture('realworld_eval_data/'+NAME)
out = cv2.VideoWriter('realworld_outputs/'+NAME,cv2.VideoWriter_fourcc('M','P','4','V'), 60, (224,224))
MODEL_PATH ='model_checkpoints/model_vit.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)


# %%
model = RoadDetector()
load_model(model, str(MODEL_PATH))
model.to(device);


#%%
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

while(cap.isOpened()):
    t = time.time()
    ret, frame = cap.read()
    if ret == True:
        overlaid, preds = model.visualize(frame, device, middles=False)
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
