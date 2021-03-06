import cv2
import os, shutil
from glob import glob
import imgaug.augmenters as iaa
from random import randrange
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import threading
from colorama import init
from time import sleep
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import warnings


class Images:
    """
    A simple datastructure that takes a path to a bunch of images of a given format.
    Looks for XML files named the same as the images for labels

    :key self.lst: List of all the images proper paths
    :key self.xml_lst: List of all the xml files paths
    :key self.size: Returns the length of self.lst
    """

    def __init__(self, path, format='JPG'):

        if type(path) != str:
            self.xml_lst = []
            for l in path:
                self.xml_lst.extend(glob(f"{l}/*.xml"))
        else:
            self.xml_lst = glob(f"{path}/*.xml")

        if len(self.xml_lst) < 1:
            warnings.warn(f"No .xml files found, assuming no bounding boxes for all images in folder {path}",
                          SyntaxWarning)
            self.xml_lst = glob(f"{path}/*.JPG")

        self.lst = [f"{i[:-4]}.{format}" for i in self.xml_lst] if len(self.xml_lst) > 0 else glob(
            f"{path}/*.{format}")  # find the path of all images in folder

        self.size = len(self.lst)

    def random(self):
        """
        :return: Random image from the image list
        """
        return self.lst[randrange(0, self.size, 1)]


def quaterSize(imgPath, savepath):
    """
    :param imgPath: Path to image
    :param savepath: Path to save new image
    :return: An image quatersize the original in openCV format
    """
    QS_img = cv2.imread(imgPath)
    QS_img_quarter = cv2.resize(QS_img, (0, 0), fx=0.25, fy=0.25)
    cv2.imwrite(f"{savepath}/{imgPath[-12:-4]}.JPG", QS_img_quarter)
    return QS_img_quarter


def xmlToTxt(xmlPath, labelsavepath, imagename, classlist, scale=1 / 4, onlybird=False):
    """
    :param xmlPath: Path to labels in XML form
    :param labelsavepath: Path to save the labels
    :param imagename: Filename of the image
    :param classlist: List of classes
    :param scale: Float: How the image was scaled compared to the original
    :param onlybird: Boolean: If to use binary class system
    :return: List: Returns the converted label data, Mainly for debugging
    """
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
            # We can now calculate the 0 - 1 data that the model wants
            width = (xmax - xmin) / imagex
            height = (ymax - ymin) / imagey
            xcenter = ((xmax + xmin) / 2) / imagex
            ycenter = ((ymax + ymin) / 2) / imagey
            classnum = 0 if onlybird else classlist.index(className)
            textfile.write(f"{classnum} {xcenter} {ycenter} {width} {height}")
            textfile.write('\n')
            data.append([classnum, xcenter, ycenter, width, height])
    return data


def RotateImageAndData(imgsource, imagename, txtdata, savepath, labelsavepath):
    """
    Data augmenter funtion, using imgaug
    :param imgsource: The image source data pref loaded using opencv
    :param imagename: Name of the image
    :param txtdata: Converted label data (from xmlToTxt)
    :param savepath: Path to save the augmented image
    :param labelsavepath: Path to save the augmented label data
    :return: array: augmented image data, list: augmented label data
    """
    # Loading
    h, w, _ = imgsource.shape
    newname = f"{imagename}{randrange(100, 999, 1)}"

    # Initialize augmentation
    # They are all chosen at random
    augmentation = iaa.Sequential([
        iaa.Sometimes(0.7, [
            iaa.Affine(rotate=(0, 359))  # Simulate rotation
        ]),
        iaa.Sometimes(0.5, [
            iaa.Fliplr(p=1.)  # Simulate different angles
        ]),
        iaa.Sometimes(0.2, [
            iaa.Flipud(p=1.)  # Simulate different angles
        ]),
        iaa.Sometimes(0.5, [
            iaa.GammaContrast(gamma=2.0)  # Simulate brightness
        ]),
        iaa.Sometimes(0.5, [
            iaa.AdditiveGaussianNoise(10, 20)  # Simulate distrubted image
        ]),
        # iaa.Sometimes(0.3, [
        #     iaa.Crop(percent=(0.10, 0.30), keep_size=True)  # Simulate sizes
        # ]),
        # iaa.Sometimes(0.1, [
        #     iaa.imgcorruptlike.Fog(severity=3)  # simulate Fog
        # ]),
        # iaa.Sometimes(0.05, [
        #     iaa.imgcorruptlike.Fog(severity=5)  # simulate even more fog
        # ]),
        # iaa.Sometimes(0.3, [
        #     iaa.AddToBrightness((-50, 50))  # Simulate more bright enviroment snow?
        # ]),
        # iaa.Sometimes(0.1, [
        #     iaa.Dropout(p=(0, 0.2))  # Simulate different noise type
        # ])
    ])

    if len(txtdata) > 0:
        # Data manipulation
        # We convert the data back to x and y coordinates
        bboxes = []
        for obj in txtdata:
            classnum, xcenter, ycenter, width, height = obj
            x1 = (xcenter - width / 2) * w
            y1 = (ycenter - height / 2) * h
            x2 = (xcenter + width / 2) * w
            y2 = (ycenter + height / 2) * h
            bboxes.append(BoundingBox(x1, y1, x2, y2, label=classnum))

        # We then calculate the new bounding boxes for the augmented image
        bbs = BoundingBoxesOnImage(bboxes, shape=imgsource.shape)
        new_image, new_bbs = augmentation(image=imgsource, bounding_boxes=bbs)
        # new_bbs = new_bbs.remove_out_of_image()

        # Data Saving
        cv2.imwrite(f"{savepath}/{newname}.JPG", new_image)

        # convert it back to the model specific data layout and save it to given path
        aug_data = []
        with open(f"{labelsavepath}/{newname}.txt", 'w') as textfile:
            for i in new_bbs.bounding_boxes:
                x1, y1, x2, y2, classnum = i.x1, i.y1, i.x2, i.y2, i.label

                awidth = (x2 - x1) / w
                aheight = (y2 - y1) / h
                axcenter = ((x2 + x1) / 2) / w
                aycenter = ((y2 + y1) / 2) / h

                textfile.write(f"{classnum} {axcenter} {aycenter} {awidth} {aheight}")
                textfile.write('\n')
                aug_data.append([classnum, axcenter, aycenter, awidth, aheight])
    else:
        warnings.warn("No bounding box data given, proceding without", UserWarning)
        new_image = augmentation(image=imgsource)
        aug_data = []

    return new_image, aug_data


def imageToGray(imgsource, imagename, txtdata, savepath, labelsavepath):
    """
    Converts image to grayscale.
    :param imgsource: Loaded image with opencv
    :param imagename: Name of image
    :param txtdata: Label data in xmlToTxt format
    :param savepath: Path to save image
    :param labelsavepath: Path to save label
    :return: None
    """
    # convert to grayscale
    gray = cv2.cvtColor(imgsource, cv2.COLOR_BGR2GRAY)
    newname = f"{imagename}{randrange(100, 999, 1)}"
    cv2.imwrite(f"{savepath}/{newname}.JPG", gray)

    # save the label to the new image
    if len(txtdata) > 0 and len(labelsavepath) > 1:
        with open(f"{labelsavepath}/{newname}.txt", 'w') as textfile:
            for obj in txtdata:
                classnum, xcenter, ycenter, width, height = obj
                textfile.write(f"{classnum} {xcenter} {ycenter} {width} {height}")
                textfile.write('\n')
    else:
        warnings.warn(f"No bounding box data or label save path was given, saveing image without {newname}",
                      UserWarning)


def testBoundingBoxes(imagepath, labelpath, N):
    """
    Draws the bounding boxes from the label on the image, saves to new folder
    :param imagepath: Path to images to test
    :param labelpath: Path to labels matching images
    :param N: How many to test, Does random sampling
    :return: Debug data, image name and label data
    """
    # Run a test on a random image
    # get the images data as our class and choose a random to test
    test = Images(imagepath, format='JPG')
    debug_data = []

    # Random sampling
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
        cv2.imwrite(f"BoundingTest/{imagename}.JPG", im)
        debug_data.append([imagename, locations])
    return debug_data


def main(position, img, savepath, labelsavepath, classlist, onlybird, N=1):
    """
    Main is the runtimer, only to be used if the file itself is getting ran. Does a sequence of procedures to generate
    an augmented data set.

    :param position: Position for tqdm, 0 if not multithreading
    :param img: Dataclass loaded with Image class
    :param savepath: Path to save images
    :param labelsavepath: Path to save Labels
    :param classlist: Class list if any, else set to None
    :param onlybird: Boolean: if True use binary class
    :param N: How many augmented copies to generate
    :return: None
    """
    for i in tqdm(range(img.size), desc=f"Generating images on thread{position}", position=position, leave=True):
        imagepath = img.lst[i]
        xmlpath = img.xml_lst[i]
        imagename = xmlpath.split("\\")[-1][:-4]
        image = quaterSize(imagepath, savepath)
        objectdata = xmlToTxt(xmlpath, labelsavepath, imagename, classlist, scale=1 / 4, onlybird=onlybird)
        imageToGray(image, imagename, objectdata, savepath, labelsavepath)

        for n in range(N):
            rot_image, rot_data = RotateImageAndData(image, imagename, objectdata, savepath, labelsavepath)

            imageToGray(rot_image, imagename, rot_data, savepath, labelsavepath)


if __name__ == "__main__":
    init()
    ## Parameters
    savepath = "OneClassDataSet/images"
    labelsavepath = "OneClassDataSet/labels"
    classlist = ['Duck', 'Cormorant', 'Heron;fa', 'Duck;fa', 'Goose', 'Cormorant;fa', 'Goose;fa', 'Swan;fa',
                 'Mute Swan', 'Seagull', 'Swan', 'Heron']

    # These are the locations we want to pull images from
    path1 = "../images/*"
    path2 = "../images2/*"
    path3 = "../images7/*"
    debugpath = "picked"
    # paths = [path1, path2]
    # paths = [path3]
    paths = [debugpath]

    # Parameters specifying runtime settings
    threads = 1
    N = 3
    onlybird = True
    delete = True  # Delete the old folders if any?
    debug = False  # wether to run the debug function

    # Delete old dataset if specified
    if delete:
        try:
            shutil.rmtree("oneClassDataSet")
            shutil.rmtree("BoundingTest")
        except:
            print()

    # Check if the folders we want to save in exist, if not create them
    os.makedirs(savepath, exist_ok=True)
    os.makedirs(labelsavepath, exist_ok=True)

    # we "load" the images we want to augment into the class
    img = Images(paths)

    # Split to threads
    N = N // threads
    # This is not the smartes multithreading, but it is faster than nothing
    # Every thread runs N/threads augment copies of each image.
    # Could be done faster by smart allocating images to threads but I can't be bothered
    if not debug:
        # Create threads
        for i in range(threads):
            locals()[f"t{i}"] = threading.Thread(target=main,
                                                 args=(i, img, savepath, labelsavepath, classlist, onlybird, N))

        for i in range(threads):
            locals()[f"t{i}"].start()

        for i in range(threads):
            locals()[f"t{i}"].join()

        # Clear the prompt because tqdm leaves a mess
        os.system("cls")
        print("Generated all images")
        print("Running bounding box check")
        sleep(0.5)
        # test some random images from path
        # Threaded because we can

        for i in range(threads):
            locals()[f"t{i}"] = threading.Thread(target=testBoundingBoxes,
                                                 args=(savepath, labelsavepath, 100))

        for i in range(threads):
            locals()[f"t{i}"].start()

        for i in range(threads):
            locals()[f"t{i}"].join()
        # os.system("cls")
        print("Bounding box generation done succesfully")
        print("Please check the BoundingBoxCheck folder for bounding box validation")
        sleep(0.5)

    else:  # This is the debugging program
        # We run only a selected few images through the system
        main(1, img, savepath, labelsavepath, classlist, onlybird, 1)
        print(testBoundingBoxes(savepath, labelsavepath, 10))
