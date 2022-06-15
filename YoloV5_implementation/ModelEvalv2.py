import os

# os.system("python yolov5/val.py --task test --img 1024 --weights last.pt --data birds.yaml")
os.system("python yolov5/val.py --img 1024 --weights last.pt --data birds.yaml --name valLast")

os.system("python yolov5/val.py --img 1024 --weights best.pt --data birds.yaml --name valBest")

# os.system("python yolov5/val.py --img 1024 --weights yolov5l.pt --data birds.yaml --name valbaseline")
