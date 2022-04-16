import cv2
import os, shutil
import glob
from PIL import Image
import random
import numpy as np
import tqdm
import threading
from colorama import init


class Images:
    def __init__(self, path):
        self.lst = glob.glob(path)
        self.size = len(self.lst)

    def random(self):
        return self.lst[random.randrange(0, self.size, 1)]


def createRandomImage(front, background, saveLocation, minAmount, maxAmount, path):
    locations = []  # getting the location for later
    size = background.size
    # Add a random amount of birds
    for i in range(random.randrange(minAmount, maxAmount, 1)):
        x = random.randrange(int(size[0] * 0.2), int(size[0] * 0.80), 1)  # find an x position
        y = random.randrange(int(size[1] * 0.2), int(size[1] * 0.80), 1)  # find a y position
        delta = random.randrange(500, 1000, 1) / 1000  # scale the bird so we have different sizes
        resize = [int(i) for i in np.multiply(front.size, delta)]  # scale the pixel size and convert to integer
        refront = front.resize(resize)  # reassign to the new scaled version
        background.paste(refront, (x, y), refront)  # add bird to image
        locations.append((x, y, resize))  # append its location
    filename = f"{path[6:-4]}{random.randrange(100000, 999999, 1)}"  # generate a random filename that we can save as
    background.save(f"{saveLocation}/{filename}.jpg")
    return locations, filename  # [x, y, resize], "file"


def getBoundingData(front, background, locations, labelpath, classlist, filename):
    class_name = filename[:-6].lower()  # Remove numbers from name should yield class_name
    xim, yim = background.size
    loclog = []
    with open(f"{labelpath}/{filename}.txt", 'w') as textfile:
        for location in locations:
            x1, y1, size = location
            x2, y2 = np.add([x1, y1], size)

            width = (x2 - x1) / xim
            height = (y2 - y1) / yim
            xcenter = ((x2 + x1) / 2) / xim
            ycenter = ((y2 + y1) / 2) / yim
            classnum = classlist.index(class_name)
            textfile.write(f"{classnum} {xcenter} {ycenter} {width} {height}")
            textfile.write('\n')
            loclog.append([x1, y1, x2, y2])

    return loclog


def testBoundingBoxes(position, datapath, labelpath):
    for _ in tqdm.tqdm(range(int(N / 100 if not N / 100 < 1 else 1)), desc=f"Running boundingBox check on {position}",
                       position=position):
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
                x1 = int(centerx - (width / 2))
                x2 = int(centerx + (width / 2))
                y1 = int(centery - (height / 2))
                y2 = int(centery + (height / 2))
                cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 1)
                locations.append([x1, y1, x2, y2])
            except:
                continue

        # Show the test image
        os.makedirs("BoundingTest", exist_ok=True)
        cv2.imwrite(f"BoundingTest/{ranimage[-14:]}", im)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return


def runtime(position, N, savelocation, labelsavepath, minbirds, maxbirds, classlist):
    # Runtime
    for _ in tqdm.tqdm(range(N), desc=f"Generating images on thread {position}", position=position):
        front_path = birds.random()
        background = backgrounds.random()
        with Image.open(front_path) as front:
            with Image.open(background) as background:
                locations, filename = createRandomImage(front, background, savelocation, minbirds, maxbirds, front_path)
                getBoundingData(front, background, locations, labelsavepath, classlist, filename)


if __name__ == "__main__":
    # Colorama init
    init()

    # This is for testing only
    try:
        shutil.rmtree("dataset")
        shutil.rmtree("BoundingTest")
    except:
        print("Nothing to remove")

    ## Parameters
    birds = Images("birds/*.png")
    backgrounds = Images("quater/*.jpg")
    savelocation = "dataset/images"
    labelsavepath = "dataset/labels"
    minbirds = 1
    maxbirds = 4
    threads = 16
    N = 100000
    classlist = ['Duck', 'Cormorant', 'Heron;fa', 'Duck;fa', 'Goose', 'Cormorant;fa', 'Goose;fa', 'Swan;fa',
                 'Mute Swan', 'Seagull', 'Swan', 'Heron']

    # Class list to lowercase
    for i in range(len(classlist)):
        classlist[i] = classlist[i].lower()

    os.makedirs(savelocation, exist_ok=True)
    os.makedirs(labelsavepath, exist_ok=True)

    # Lets make some threads:
    N = int(N / threads)
    threads = threads - 1
    for i in range(threads):
        locals()[f"t{i}"] = threading.Thread(target=runtime,
                                             args=(i, N, savelocation, labelsavepath, minbirds, maxbirds, classlist))

    for i in range(threads):
        locals()[f"t{i}"].start()

    for i in range(threads):
        locals()[f"t{i}"].join()

    # test some random images from path
    # Threaded because we can
    for i in range(threads):
        locals()[f"t{i}"] = threading.Thread(target=testBoundingBoxes, args=(
            i, "dataset/images/*.jpg", "dataset/labels"))

    for i in range(threads):
        locals()[f"t{i}"].start()

    for i in range(threads):
        locals()[f"t{i}"].join()
