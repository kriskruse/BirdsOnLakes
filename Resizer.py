from PIL import Image
import os
import cv2
import numpy as np
import glob

# target_image = cv2.imread('images/1/IMAG0030.JPG')
new_image_height, new_image_width, _ = 1200, 1200, 0
color = (0, 0, 0)
size = (new_image_height,new_image_width)
folder_pat = "Kaggle_MOdel/padded/train/*/*"

for image_path in glob.glob(folder_pat):

    with Image.open(image_path) as im:

        im.thumbnail(size, Image.ANTIALIAS)

        _, _, new_image_path = image_path.split("/")
        dir_path = f"Kaggle_MOdel/pad_1220/{new_image_path}"
        os.makedirs(dir_path[0:-8], exist_ok=True)
        im.save(f"Kaggle_MOdel/pad_1220/{new_image_path}", "JPEG")

