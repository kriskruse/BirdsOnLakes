import cv2
import glob
from bs4 import BeautifulSoup

paths = ["../quartersize_images/*.xml"]

for path in paths:
    for xmlpath in glob.glob(path):
        file = open(xmlpath, 'r')
        data = file.read()
        bs_data = BeautifulSoup(data, "xml")

        xmin = int(int(bs_data.contents[0].contents[13].contents[9].contents[1].contents[0]) / 4)
        ymin = int(int(bs_data.contents[0].contents[13].contents[9].contents[3].contents[0]) / 4)
        xmax = int(int(bs_data.contents[0].contents[13].contents[9].contents[5].contents[0]) / 4)
        ymax = int(int(bs_data.contents[0].contents[13].contents[9].contents[7].contents[0]) / 4)

        img_path = f"{xmlpath[:-4]}.JPG"
        img = cv2.imread(img_path)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        cv2.imshow("Show", img)
        cv2.waitKey()
