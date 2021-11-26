import time
import numpy as np
import sounddevice as sd
from flask import Flask
import threading

sd.default.device = "Yeti Nano"

def soundProcess():
    duration = 3
    while True:
        def print_sound(indata, outdata, frames, time, status):
            volume_norm = np.linalg.norm(indata)*1000
            print("|" * int(volume_norm))
            outdata[:] = indata * 1
        with sd.Stream(callback=print_sound):
            sd.sleep(duration * 1000)


app = Flask(__name__)
@app.route('/')
def home():
    return 'Hello, World!'

soundThread = threading.Thread(target=soundProcess)
soundThread.start()

app.run(host='0.0.0.0', port=443)
