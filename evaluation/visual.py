import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image


def get_frames(f):
    print(f)
    pil_img = Image.open(f)
    img = np.array(pil_img)
    return img

def play_sequence(files):
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, get_frames, frames=files, interval=50, blit=True)
    plt.show()
    input('Press any key to quit sequence')


def play_matches(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    play_sequence(files)
