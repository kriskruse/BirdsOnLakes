import os

os.system("python yolov5/val.py --task test --img 1024 --weights last.pt --data ../birds.yaml")
