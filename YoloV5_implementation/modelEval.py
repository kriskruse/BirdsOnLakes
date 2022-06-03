import torch
from PIL import Image
import os
from glob import glob
# Set the working path to yolov5 repo
os.chdir("/Users\KrisK\Desktop\Github\BirdsOnLakes\YoloV5_implementation\yolov5")

# Model, path to yolo folder, custom, path to weights, source=local
model = torch.hub.load('', 'custom', path='../best.pt', source='local')

# get images to run on
paths = glob("C:/Users\KrisK\Desktop\Github\BirdsOnLakes/Non_labeled_images\images picked/*.JPG")
images = []
for path in paths:
    images.append(Image.open(path))
# im1 = "../OneClassTestSet/images/test/IMAG0199.JPG"
# im1 = Image.open(im1)

# inference
results = model(images, size=1008)

# Results
results.print()
os.makedirs("../testresults", exist_ok=True)
results.show()

print(results.pandas().xyxy)  # im1 predictions (pandas)
# print(results.pandas().xyxy[1])
# print(results.pandas().xyxy[2])