import cv2
import os, shutil
from glob import glob
from PIL import Image, ImageEnhance
from random import randrange, choice
from numpy import add, multiply as npadd, npmultiply
import tqdm
import threading
from colorama import init
from time import sleep


class Images:
    def __init__(self, path):
        self.lst = glob(path)
        self.size = len(self.lst)

    def random(self):
        return self.lst[randrange(0, self.size, 1)]


def createRandomImage(front, background, saveLocation, minAmount, maxAmount, path, usefilter=False, darken=None):
    locations = []  # getting the location for later
    size = background.size
    # Add a random amount of birds
    for i in range(randrange(minAmount, maxAmount, 1)):
        x = randrange(int(size[0] * 0.2), int(size[0] * 0.80), 1)  # find an x position
        y = randrange(int(size[1] * 0.2), int(size[1] * 0.80), 1)  # find a y position
        delta = randrange(100, 1000, 1) / 1000  # scale the bird so we have different sizes
        resize = [int(x) for x in npmultiply(front.size, delta)]  # scale the pixel size and convert to integer
        refront = front.resize(resize)  # reassign to the new scaled version

        if usefilter:
            dominantColor = get_dominant_color(background)  # find the color that dominates the image
            colorfilter = Image.new('RGBA', refront.size, color=dominantColor)  # make an image of that color
            mask = refront.convert('L')  # create a greyscale mask of the front image, so we only apply filter to bird
            refront = Image.composite(colorfilter, refront, mask)  # apply the filter
            refront.convert('RGB')  # convert to RBG to eliminate any alpha values the bird might have

        if darken is not None:  # Try to make the lighting of the bird more random
            enhancer = ImageEnhance.Brightness(refront)
            refront = enhancer.enhance(darken)

        background.paste(refront, (x, y), refront)  # add bird to image
        locations.append((x, y, resize))  # append its location
    filename = f"{path[6:-4]}{randrange(100000, 999999, 1)}"  # generate a random filename that we can save as
    background.save(f"{saveLocation}/{filename}.jpg")
    return locations, filename  # [x, y, resize], "file"


def getBoundingData(front, background, locations, labelpath, classlist, filename):
    class_name = filename[:-6].lower()  # Remove numbers from name should yield class_name
    xim, yim = background.size  # get the size of the background
    loclog = []
    # we create a text file with a name matching the image name
    with open(f"{labelpath}/{filename}.txt", 'w') as textfile:
        for location in locations:
            x1, y1, size = location  # get the top left corner position
            x2, y2 = npadd([x1, y1], size)  # calculate the bottom right

            # We then convert the xs and ys to the 4 sizes specified in the YoloV5 model description
            width = (x2 - x1) / xim
            height = (y2 - y1) / yim
            xcenter = ((x2 + x1) / 2) / xim
            ycenter = ((y2 + y1) / 2) / yim
            classnum = classlist.index(class_name)
            textfile.write(f"{classnum} {xcenter} {ycenter} {width} {height}")  # save everything to file
            textfile.write('\n')  # newline in case there is more than one bird
            loclog.append([x1, y1, x2, y2])  # log position data, in case of debugging

    return loclog  # only for debugging


# https://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image
def get_dominant_color(pil_img, palette_size=16):
    # Resize image to speed up processing
    img = pil_img.copy()
    img.thumbnail((100, 100))

    # Reduce colors (uses k-means internally)
    paletted = img.convert('P', palette=Image.Palette, colors=palette_size)

    # Find the color that occurs most often
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    palette_index = color_counts[0][1]
    dominant_color = palette[palette_index * 3:palette_index * 3 + 3]

    return tuple(dominant_color)


def testBoundingBoxes(position, N, datapath, labelpath):
    for _ in tqdm.tqdm(range(int(N / 1000 if not N / 1000 < 1 else 1)),
                       desc=f"Running boundingBox check on thread{position}",
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

        # Save the test images to a folder
        os.makedirs("BoundingTest", exist_ok=True)
        cv2.imwrite(f"BoundingTest/{ranimage[15:]}", im)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return


def runtime(position, N, savelocation, labelsavepath, minbirds, maxbirds, classlist, ranfilter=False, darkenbird=False):
    # Runtime
    for _ in tqdm.tqdm(range(N), desc=f"Generating images on thread{position}", position=position):
        front_path = birds.random()  # get a random bird
        background = backgrounds.random()  # get a random background
        # load the images into memory
        with Image.open(front_path) as front:
            with Image.open(background) as background:
                darkenvalue = randrange(50, 100, 1) / 100 if darkenbird else None  # Random how strong the darkening is
                usefilter = random.choice([True, False]) if ranfilter else False  # If the filter is on we randomize it
                # create the new image and the mathing boundingbox data
                locations, filename = createRandomImage(front, background, savelocation, minbirds, maxbirds, front_path,
                                                        usefilter, darkenvalue)
                getBoundingData(front, background, locations, labelsavepath, classlist, filename)


if __name__ == "__main__":
    # Colorama init
    init()

    # This is for testing only
    try:
        shutil.rmtree("dataset")
        shutil.rmtree("BoundingTest")
    except:
        print("")

    # Parameters
    birds = Images("birds/*.png")
    backgrounds = Images("quater/*.jpg")
    savelocation = "dataset/images"
    labelsavepath = "dataset/labels"
    minbirds = 1
    maxbirds = 4
    usefilter = True
    darkenbird = True
    threads = 16
    N = 20
    classlist = ['Duck', 'Cormorant', 'Heron;fa', 'Duck;fa', 'Goose', 'Cormorant;fa', 'Goose;fa', 'Swan;fa',
                 'Mute Swan', 'Seagull', 'Swan', 'Heron']

    # Class list to lowercase
    for i in range(len(classlist)):
        classlist[i] = classlist[i].lower()

    os.makedirs(savelocation, exist_ok=True)
    os.makedirs(labelsavepath, exist_ok=True)

    # Lets make some threads:
    N = int(N / threads)  # splits the load evenly on the threads
    threads = threads - 1  # Just an adjustment so it matches the expected with the for loop
    for i in range(threads):
        locals()[f"t{i}"] = threading.Thread(target=runtime,
                                             args=(i, N, savelocation, labelsavepath, minbirds, maxbirds, classlist,
                                                   usefilter, darkenbird))

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
        locals()[f"t{i}"] = threading.Thread(target=testBoundingBoxes, args=(
            i, N, "dataset/images/*.jpg", "dataset/labels"))

    for i in range(threads):
        locals()[f"t{i}"].start()

    for i in range(threads):
        locals()[f"t{i}"].join()
    os.system("cls")
    print("Bounding box generation done succesfully")
    print("Please check the BoundingBoxCheck folder for bounding box validation")
    sleep(0.5)
