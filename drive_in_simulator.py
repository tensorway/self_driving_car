'''
a script adapted from
https://github.com/udacity/CarND-Behavioral-Cloning-P3
used to test the driving in the simulator
'''
#%%
import argparse
import base64
from datetime import datetime
import os
import shutil
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from driving_agents import CenterOfMassFollower as Agent
import logging
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using" + str(device))

# logging.basicConfig(
#     format='%(asctime)s:%(levelname)s:%(message)s',
#     filename='simulator_driving.log', 
#     encoding='utf8', 
#     level=logging.DEBUG)

#%%
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        try:
            # The current image from the center camera of the car
            imgString = data["image"]
            image = Image.open(BytesIO(base64.b64decode(imgString)))
            steering_angle = agent.predict(image)
            speed = data["speed"]
            throttle = 2 * (10-float(speed))/10
            angl = data["steering_angle"]
            # print(f"predicted_sangle={steering_angle:5.4f}      curr_angle={angl:5.4f}        throttle={throttle:5.4f}")
            print(angl, steering_angle, throttle)
            send_control(steering_angle, throttle)
        except Exception as e:
            print(str(e))

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect " + str(sid))
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    agent = Agent(device, 1/100*3.14)

    if args.image_folder != '':
        logging.info("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        logging.info("RECORDING THIS RUN ...")
    else:
        logging.info("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

# %%
