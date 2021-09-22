#%%
import serial
import time
import subprocess
import logging

class Controller:

    def __init__(self) -> None:
        self.open()

    def open(self):
        self.serial = serial.Serial(self.get_device(), 115200)
        # wakeup grbl
        self.serial.write("\r\n\r\n".encode())
        # Wait for grbl to initialize 
        time.sleep(2)  

    def get_device(self):
        '''
        get the serial port where 
        something is connected
        '''
        result = subprocess.check_output("ls -l /dev/ttyUSB*", shell=True, universal_newlines=True, stderr=subprocess.DEVNULL)
        device = result.replace('\n', '').split(' ')[-1]
        return device

    def move(self, x=None, y=None, speed=300, debug=False):
        # Stream g-code to grbl
        if x is None and y is None:
            return
        X = f'X{x}' if x is not None else ''
        Y = f'Y{y}' if y is not None else ''
        line = f"G1 {X} {Y} F{speed}" #XYgoto Fspeed
        # line = "G92 X0 Y0" # zero
        return self.send(line)

    def home(self):
        self.send("G28")

    def zero(self):
        self.send("G92 X0 Y0")

    def send(self, line):
        logging.debug('Sending: ' + line),
        # Send g-code block to grbl
        self.serial.write((line + '\r\n').encode()) 

        # Wait for grbl response with carriage return
        grbl_out = self.serial.readline().decode("utf-8").strip()
        logging.info("grbl returned:"+ grbl_out)
        logging.debug (' : ' + grbl_out)
        return grbl_out

    def close(self):
        self.serial.close()
    

# %%
if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
        filename='example.log', 
        encoding='utf8', 
        level=logging.DEBUG)
    controller = Controller()
    controller.zero()
    print(controller.move(15, 10))

# %%
