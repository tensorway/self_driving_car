#%%
import serial
import time
import subprocess
#%%

# Open grbl serial port
def get_device():
    result = subprocess.check_output("ls -l /dev/ttyUSB*", shell=True, universal_newlines=True, stderr=subprocess.DEVNULL)
    device = result.replace('\n', '').split(' ')[-1]
    return device
    
s = serial.Serial(get_device(),115200)

# time.sleep(2)   # Wait for grbl to initialize 
# Wake up grbl
s.write("\r\n\r\n".encode())
time.sleep(2)   # Wait for grbl to initialize 
s.flushInput()  # Flush startup text in serial input
#%%
# Stream g-code to grbl
line = "G1 X2 Y3 F500"
l = line.strip() # Strip all EOL characters for consistency
print ('Sending: ' + l),
s.write((l + '\n').encode()) # Send g-code block to grbl
grbl_out = s.readline() # Wait for grbl response with carriage return
print (' : ' + grbl_out.decode("utf-8").strip())
#%%

# Close file and serial port
s.close()    
#%%
# %%
get_device()