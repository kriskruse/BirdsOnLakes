from glob import glob
from os import listdir
from PIL import Image
import cv2

broken = []
once = False
for filename in listdir('./OneClassBetterAugment2/images/'):
    if filename.endswith('.JPG'):
        if not once:
            print("Found images", filename)
            once = True
        try:
            img = cv2.imread('./oneClassDataSet/images/' + filename)  # open the image file
            img = cv2.resize(img, (0,0),fx=1/4, fy=1/4 )  # verify that it is, in fact an image
            # print("is good:", filename)
        except Exception as e:
            # print('Bad file:', filename)  # print out the names of corrupt files
            # print(e)
            broken.append(filename)

print(broken)
print(len(broken))