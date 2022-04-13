import cv2
import os, shutil
import glob
from PIL import Image
import random
import numpy as np


class Images:
    def __init__(self, path):
        self.lst = glob.glob(path)
        self.size = len(self.lst)

    def random(self):
        return self.lst[random.randrange(0, self.size, 1)]


def createRandomImage(background, front, saveLocation, minAmount, maxAmount, path):
    locations = []  # getting the location for later
    size = background.size
    # Add a random amount of birds
    for i in range(random.randrange(minAmount, maxAmount, 1)):
        x = random.randrange(int(size[0] * 0.2), int(size[0] * 0.80), 1)  # find an x position
        y = random.randrange(int(size[1] * 0.2), int(size[1] * 0.80), 1)  # find a y position
        delta = random.randrange(500, 1000, 1) / 1000  # scale the bird so we have different sizes
        resize = [int(i) for i in np.multiply(front.size, delta)]  # scale the pixel size and convert to integer
        front = front.resize(resize)  # reassign to the new scaled version
        background.paste(front, (x, y), front)  # add bird to image
        locations.append((x, y, delta))  # append its location
    filename = f"{path[6:-4]}{random.randrange(100000, 999999, 1)}"  # generate a random filename that we can save as
    background.save(f"{saveLocation}/{filename}.jpg")
    return locations, filename  # [x, y, delta], "file"


def getBoundingData(front, background, locations, labelpath, classlist, filename):
    class_name = filename[:-6].lower()  # Remove numbers from name should yield class_name
    xim, yim = background.size
    loclog = []
    with open(f"{labelpath}/{filename}.txt", 'w') as textfile:
        for location in locations:
            x1, y1, delta = location
            resize = [int(i) for i in np.multiply(front.size, delta)]
            x2, y2 = np.add([x1, y1], resize)

            width = (x2 - x1) / xim
            height = (y2 - y1) / yim
            xcenter = ((x2 + x1) / 2) / xim
            ycenter = ((y2 + y1) / 2) / yim
            classnum = classlist.index(class_name)
            textfile.write(f"{classnum} {xcenter} {ycenter} {width} {height}")
            textfile.write('\n')
            loclog.append([x1, x2, y1, y2])

    return loclog


def testBoundingBoxes(datapath, labelpath):
    # Run a test on a random image
    # get the images data as our class and choose a random to test
    test = Images(datapath)
    ranimage = test.random()
    im = cv2.imread(ranimage)

    # find and load the label data
    with open(f"{labelpath}/{ranimage[15:-4]}.txt") as textfile:
        data = textfile.read().split("\n")
        data = [i.split() for i in data]  # [[],[]]
        data = [[float(j) for j in i] for i in data]

    # Calculate and draw the bounding boxes
    locations = []
    for dat in data:
        try:
            _, centerx, centery, width, height = dat
            imy, imx, _ = im.shape
            centerx = centerx * imx
            centery = centery * imy
            width = width * imx
            height = height * imy
            x1 = int(centerx - width / 2)
            x2 = int(centerx + width / 2)
            y1 = int(centery - height / 2)
            y2 = int(centery + height / 2)
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 1)
            locations.append([x1, x2, y1, y2])
        except:
            continue

    # Show the test image
    cv2.imshow("show", im)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return locations


if __name__ == "__main__":

    ## Parameters
    birds = Images("birds/*.png")
    backgrounds = Images("quater/*.jpg")
    savelocation = "dataset/images"
    labelsavepath = "dataset/labels"
    minbirds = 3
    maxbirds = 4
    N = 1
    classlist = ['Duck', 'Cormorant', 'Heron;fa', 'Duck;fa', 'Goose', 'Cormorant;fa', 'Goose;fa', 'Swan;fa',
                 'Mute Swan', 'Seagull', 'Swan', 'Heron']

    # Class list to lowercase
    for i in range(len(classlist)):
        classlist[i] = classlist[i].lower()

    os.makedirs(savelocation, exist_ok=True)
    os.makedirs(labelsavepath, exist_ok=True)

    # Runtime
    for i in range(N):
        front_path = birds.random()
        background = backgrounds.random()
        with Image.open(front_path) as front:
            with Image.open(background) as background:
                locations, filename = createRandomImage(background, front, savelocation, minbirds, maxbirds, front_path)
                boundloc = getBoundingData(front, background, locations, labelsavepath, classlist, filename)

    # test a random image from path
    for i in range(1):
        testloc = testBoundingBoxes("dataset/images/*.jpg", "dataset/labels")
        print(boundloc)
        print(testloc)
