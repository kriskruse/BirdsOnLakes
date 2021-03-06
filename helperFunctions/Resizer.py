import shutil
from PIL import Image
import os
import cv2
import numpy as np
import glob


# paths = ["../images/*/*.txt", "../images2/*/*.txt"]
# savepath = "../quatersize_images"
paths = ["backgrounds/*.JPG"]
savepath = "quater"

os.makedirs(savepath,exist_ok=True)

for path in paths:
    for file in glob.glob(path):
        img = cv2.imread(f"{file[:-4]}.JPG")
        img_quarter = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
        cv2.imwrite(f"{savepath}/{file[-12:-4]}.JPG", img_quarter)
        #shutil.copy(f"{file[:-4]}.xml", f"../quartersize_images/")

