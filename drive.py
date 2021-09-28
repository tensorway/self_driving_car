#%%
import time
import numpy as np
import cv2
import torch
import random
import logging
import threading
from goprocam import GoProCamera
from actuation import Controller
from driving_agents import CenterOfMassFollower as Agent

MOVE_COEFF = 3
VIDEO_NAME = str(random.randint(0, 10**6))+'.mp4'

logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    filename='driving.log', 
    # encoding='utf8', 
    level=logging.DEBUG)


# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()
        self.run = True
        self.last = np.zeros((224, 224, 3))

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            ret = self.cap.grab()
            if not ret:
                break

    # retrieve latest frame
    def read(self):
        try:
            self.last = self.cap.retrieve()
        except Exception as e:
            print(e)
        return self.last

    def release(self):
        self.run = False
        time.sleep(1)
        self.cap.release()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using" + str(device))
#%%

agent = Agent(device, 1/100*3.14)
controller = Controller()
controller.zero()

#%%
gpCam = GoProCamera.GoPro()
cap = VideoCapture("udp://127.0.0.1:10000")
out = cv2.VideoWriter('driving_outputs/'+VIDEO_NAME,cv2.VideoWriter_fourcc('M','P','4','V'), 60, (224,224))
time.sleep(1)

#%%
while True:
    t = time.time()
    ret, frame = cap.read()
    try:
        steering_angle, overlaid = agent.predict(frame, visualize=True)
    except Exception as e:
        print(e)
        time.sleep(0.1)
        continue
    controller.move(steering_angle*MOVE_COEFF, steering_angle*MOVE_COEFF)
    cv2.imshow("overlaid", cv2.resize(overlaid, (512, 512)))
    out.write(overlaid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(f"fps={1/(time.time()-t)}     overlaid.shape={overlaid.shape}")

out.release()
cap.release()
cv2.destroyAllWindows()



# %%
