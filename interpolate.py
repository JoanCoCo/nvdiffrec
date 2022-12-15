import os
import numpy as np
from PIL import Image

from interpolation import interpolation_utils

SOURCE_FOLDER = "data/sketch/boy"
NAME_PREFIX = "char_"
EXTENSION = ".png"
NUMBER_OF_IMAGES = 9
RESULTS_FOLDER = "test_interp"

if __name__ == "__main__":
    img_seq = []
    for i in range(0, NUMBER_OF_IMAGES):
        img = Image.open(os.path.join(SOURCE_FOLDER, NAME_PREFIX + str(i) + EXTENSION))
        img = img.resize((256, 1024), Image.ANTIALIAS)
        img = img.convert("RGB")
        img_seq += [np.array(img)]
    smooth_seq = interpolation_utils.expand_sequence(img_seq, intermediante_frames=4)
    for i, frame in enumerate(smooth_seq):
        Image.fromarray(frame).resize((392, 1153), Image.ANTIALIAS).save(os.path.join(RESULTS_FOLDER, "frame{:d}.png".format(i)))
