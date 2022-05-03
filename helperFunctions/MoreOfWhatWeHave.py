import random

import cv2
import os, shutil
from glob import glob
from affine import Affine

import numpy as np
from PIL import Image, ImageEnhance
from random import randrange, choice
from bs4 import BeautifulSoup
from numpy import add as npadd, multiply as npmultiply
import tqdm
import threading
from colorama import init
from time import sleep


class Images:
    def __init__(self, path, format='jpg'):

        if type(path) != str:
            self.xml_lst = []
            for l in path:
                self.xml_lst.extend(glob(f"{l}/*.xml"))
        else:
            self.xml_lst = glob(f"{path}/*.xml")

        self.lst = [f"{i[:-4]}.{format}" for i in self.xml_lst] if len(self.xml_lst) > 0 else glob(
            f"{path}/*.{format}")  # find the path of all images in folder
        self.size = len(self.lst)

    def random(self):
        return self.lst[randrange(0, self.size, 1)]


def quaterSize(imgPath, savepath):
    QS_img = cv2.imread(imgPath)
    QS_img_quarter = cv2.resize(QS_img, (0, 0), fx=0.25, fy=0.25)
    cv2.imwrite(f"{savepath}/{imgPath[-12:-4]}.JPG", QS_img_quarter)
    return QS_img_quarter


def xmlToTxt(xmlPath, labelsavepath, imagename, classlist, scale=1 / 4, onlybird=False):
    if classlist is None:
        classlist = ['Duck', 'Cormorant', 'Heron;fa', 'Duck;fa', 'Goose', 'Cormorant;fa', 'Goose;fa', 'Swan;fa',
                     'Mute Swan', 'Seagull', 'Swan', 'Heron']

    file = open(xmlPath, 'r')
    data = file.read()
    bs_data = BeautifulSoup(data, "xml")
    data = []

    with open(f"{labelsavepath}/{imagename}.txt", 'w') as textfile:
        for i in range(13, 25, 2):  # This is very specific locations for our XML structure only
            try:
                className = bs_data.contents[0].contents[i].contents[1].contents[0]
                imagex = int(bs_data.contents[0].contents[9].contents[1].contents[0]) * scale
                imagey = int(bs_data.contents[0].contents[9].contents[3].contents[0]) * scale
                xmin = int(bs_data.contents[0].contents[i].contents[9].contents[1].contents[0]) * scale
                ymin = int(bs_data.contents[0].contents[i].contents[9].contents[3].contents[0]) * scale
                xmax = int(bs_data.contents[0].contents[i].contents[9].contents[5].contents[0]) * scale
                ymax = int(bs_data.contents[0].contents[i].contents[9].contents[7].contents[0]) * scale
            except:
                break

            width = (xmax - xmin) / imagex
            height = (ymax - ymin) / imagey
            xcenter = ((xmax + xmin) / 2) / imagex
            ycenter = ((ymax + ymin) / 2) / imagey
            classnum = 1 if onlybird else classlist.index(className)
            textfile.write(f"{classnum} {xcenter} {ycenter} {width} {height}")
            textfile.write('\n')
            data.append([classnum, xcenter, ycenter, width, height])
    return data


def RotateImageAndData(imgsource, imagename, txtdata, savepath, labelsavepath):
    h, w = imgsource.shape[:2]
    cx, cy = (w / 2, h / 2)
    angle = random.randint(1, 359)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    rot_image = cv2.warpAffine(imgsource, M, (w, h))
    newname = f"{imagename}{randrange(100, 999, 1)}"
    cv2.imwrite(f"{savepath}/{newname}.jpg", rot_image)

    rot_data = []
    with open(f"{labelsavepath}/{newname}.txt", 'w') as textfile:
        for obj in txtdata:
            classnum, xcenter, ycenter, width, height = obj
            x1 = (xcenter - (width / 2)) * w
            x2 = (xcenter + (width / 2)) * w
            y1 = (ycenter - (height / 2)) * h
            y2 = (ycenter + (height / 2)) * h
            p1 = (x1, y1)
            p2 = (x2, y2)
            rp1 = Affine.rotation(-angle, (cx, cy)) * p1
            rp2 = Affine.rotation(-angle, (cx, cy)) * p2
            bcx = rp1[0] - rp2[0]
            bcy = rp1[1] - rp2[1]
            rx1, ry1 = Affine.rotation(-angle, (bcx,bcy)) * p1
            rx2, ry2 = Affine.rotation(-angle, (bcx,bcy)) * p2

            rwidth = (abs(rx2 - rx1)) / w
            rheight = (abs(ry2 - ry2)) / h
            rxcenter = (min(rx1, rx2) + rwidth / 2) / w
            rycenter = (min(ry1, ry2) + rheight / 2) / h

            textfile.write(f"{classnum} {rxcenter} {rycenter} {rwidth} {rheight}")
            textfile.write('\n')
            rot_data.append([classnum, rxcenter, rycenter, rwidth, rheight])
    return rot_image, rot_data


def imageToGray(imgsource, imagename, txtdata, savepath, labelsavepath):
    gray = cv2.cvtColor(imgsource, cv2.COLOR_BGR2GRAY)
    newname = f"{imagename}{randrange(100, 999, 1)}"
    cv2.imwrite(f"{savepath}/{newname}.jpg", gray)

    with open(f"{labelsavepath}/{newname}.txt", 'w') as textfile:
        for obj in txtdata:
            classnum, xcenter, ycenter, width, height = obj
            textfile.write(f"{classnum} {xcenter} {ycenter} {width} {height}")
            textfile.write('\n')


def testBoundingBoxes(imagepath, labelpath, N):
    # Run a test on a random image
    # get the images data as our class and choose a random to test
    test = Images(imagepath, format='jpg')
    debug_data = []

    for i in range(N):
        ranimage = test.random()
        im = cv2.imread(ranimage)
        imagename = ranimage.split('\\')[-1][:-4]

        # find and load the label data
        with open(f"{labelpath}/{imagename}.txt") as textfile:
            data = textfile.read().split("\n")
            data = [i.split() for i in data]  # [[],[]]
            data = [[float(j) for j in i] for i in data]

        # Calculate and draw the bounding boxes
        locations = []
        for dat in data:
            try:
                _, centerx, centery, width, height = dat
                imy, imx, _ = im.shape
                cx = centerx * imx
                cy = centery * imy
                w = width * imx
                h = height * imy
                x1 = int(cx - (w / 2))
                x2 = int(cx + (w / 2))
                y1 = int(cy - (h / 2))
                y2 = int(cy + (h / 2))
                print("y is wrong" if y1 == y2 else "")
                cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 1)
                locations.append([x1, y1, x2, y2])
            except:
                continue

        # Save the test images to a folder
        os.makedirs("BoundingTest", exist_ok=True)
        cv2.imwrite(f"BoundingTest/{imagename}.jpg", im)
        debug_data.append([imagename, locations])
    return debug_data


def main(position, img, savepath, labelsavepath, classlist, onlybird, N=1):
    for i in tqdm.tqdm(range(img.size), desc=f"Generating images on thread{position}", position=position):
        imagepath = img.lst[i]
        xmlpath = img.xml_lst[i]
        imagename = xmlpath.split("\\")[-1][:-4]
        image = quaterSize(imagepath, savepath)
        objectdata = xmlToTxt(xmlpath, labelsavepath, imagename, classlist, scale=1 / 4, onlybird=onlybird)
        imageToGray(image, imagename, objectdata, savepath, labelsavepath)

        for i in range(N):
            rot_image, rot_data = RotateImageAndData(image, imagename, objectdata, savepath, labelsavepath)
            imageToGray(rot_image, imagename, rot_data, savepath, labelsavepath)


if __name__ == "__main__":
    init()
    ## Parameters
    savepath = "test/images"
    labelsavepath = "test/labels"
    classlist = ['Duck', 'Cormorant', 'Heron;fa', 'Duck;fa', 'Goose', 'Cormorant;fa', 'Goose;fa', 'Swan;fa',
                 'Mute Swan', 'Seagull', 'Swan', 'Heron']
    path1 = "../images/*"
    path2 = "../images2/*"
    debugpath = "picked"
    # paths = [path1, path2]
    paths = [debugpath]
    N = 1
    threads = 12
    onlybird = False
    delete = True
    debug = True

    if delete:
        try:
            shutil.rmtree("test")
            shutil.rmtree("BoundingTest")
        except:
            print()

    os.makedirs(savepath, exist_ok=True)
    os.makedirs(labelsavepath, exist_ok=True)

    img = Images(paths)
    # print(img.xml_lst)
    # print(img.size)
    # print(img.lst)

    # main(position, img, savepath, labelsavepath, classlist, onlybird, N)

    if not debug:
        for i in range(threads):
            locals()[f"t{i}"] = threading.Thread(target=main,
                                                 args=(i, img, savepath, labelsavepath, classlist, onlybird, N))

        for i in range(threads):
            locals()[f"t{i}"].start()

        for i in range(threads):
            locals()[f"t{i}"].join()

        os.system("cls")
        print("Generated all images")
        print("Running bounding box check")
        sleep(0.5)
        # test some random images from path
        # Threaded because we can

        for i in range(threads):
            locals()[f"t{i}"] = threading.Thread(target=testBoundingBoxes,
                                                 args=(f"{savepath}/*.jpg", labelsavepath, 100))

        for i in range(threads):
            locals()[f"t{i}"].start()

        for i in range(threads):
            locals()[f"t{i}"].join()
        # os.system("cls")
        print("Bounding box generation done succesfully")
        print("Please check the BoundingBoxCheck folder for bounding box validation")
        sleep(0.5)

    else:
        main(1, img, savepath, labelsavepath, classlist, onlybird, 1)
        # TODO: Fix bounding boxes
        print(testBoundingBoxes(savepath, labelsavepath, 10))

    # make x amount of copies with different changes
    # - image shifted right or left
    # - Rotated at different angles
    # - Color-modes, maybe greyscale?
    # Combination of above.
