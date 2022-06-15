import torch
from PIL import Image
import os
from glob import glob
# Set the working path to yolov5 repo
os.chdir("/Users\KrisK\Desktop\Github\BirdsOnLakes\YoloV5_implementation\yolov5")

# Model, path to yolo folder, custom, path to weights, source=local
mod_best = torch.hub.load('', 'custom', path='../best.pt', source='local')
mod_last = torch.hub.load('', 'custom', path='../last.pt', source='local')

# get images to run on
path = glob("C:/Users\KrisK\Desktop\Github\BirdsOnLakes/Non_labeled_images\images picked/*.JPG")
path_org = glob("C:/Users\KrisK\Desktop\Github\BirdsOnLakes/Non_labeled_images\images picked/Originals/*.JPG")
images = []
img_org = []
for p in path:
    images.append(Image.open(p))
for p in path_org:
    img_org.append(Image.open(p))

# inference
res_best = mod_best(images, size=1008)
# res_last = mod_last(images, size=1008)
# res_best_B = mod_best(img_org, size=4032)
# res_last_B = mod_last(img_org, size=4032)

# Results
res_best.print()
# os.makedirs("../testresults", exist_ok=True)
res_best.show()

# res_last.print()
# res_last.show()

print(res_best.pandas().xyxy)  # im1 predictions (pandas)
# print(res_last.pandas().xyxy)
# print(results.pandas().xyxy[1])
# print(results.pandas().xyxy[2])