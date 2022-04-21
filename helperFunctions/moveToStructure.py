import os, shutil
from bs4 import BeautifulSoup
import glob
import random
import time


def imagesToPath(paths=None):
    if paths is None:
        paths = ["../images/*/*.txt", "../images2/*/*.txt"]
    if type(paths) is not list:
        paths = [paths]

    try:
        os.makedirs("../Structure/images/train")
        os.makedirs("../Structure/labels/train")
        os.makedirs("../Structure/images/test")
        os.makedirs("../Structure/labels/test")
    except:
        print("Directories exist")

    for path in paths:
        for file in glob.glob(path):
            # Get the filename of the image file
            image_name = f"{file[-12:-4]}.JPG"
            image_path = f"{file[:-12]}{image_name}"

            # randomly move the images and xml files to either test or train
            if random.random() <= 0.15:
                shutil.copy(image_path, "../Structure/images/test")
                shutil.copy(file, "../Structure/labels/test")
            else:
                shutil.copy(image_path, "../Structure/images/train")
                shutil.copy(file, "../Structure/labels/train")


def LableToStruc(paths=None, classlist=None, scale=1.0, onlybird=False):
    if classlist is None:
        classlist = ['Duck', 'Cormorant', 'Heron;fa', 'Duck;fa', 'Goose', 'Cormorant;fa', 'Goose;fa', 'Swan;fa',
                     'Mute Swan', 'Seagull', 'Swan', 'Heron']

    if paths is None:
        paths = ["../images/*/*.xml", "../images2/*/*.xml"]
    if type(paths) is not list:
        paths = [paths]

    for path in paths:
        for xmlpath in glob.glob(path):  # loop over every element in the path that is a .xml

            # Open and read the xml file
            file = open(xmlpath, 'r')
            data = file.read()
            bs_data = BeautifulSoup(data, "xml")

            # Find and add all the names/classes to the list,
            # we check if the amount of found names/classes follows an expected value
            # The expected value is found with the logic that the data placement is 13, 13+2, 13+2+2, ..
            found = []
            with open(f"{xmlpath[:-4]}.txt", 'w') as textfile:
                for i in range(13, 25, 2):
                    try:
                        className = bs_data.contents[0].contents[i].contents[1].contents[0]
                        imagex = int(bs_data.contents[0].contents[9].contents[1].contents[0])*scale
                        imagey = int(bs_data.contents[0].contents[9].contents[3].contents[0])*scale
                        xmin = int(bs_data.contents[0].contents[i].contents[9].contents[1].contents[0])*scale
                        ymin = int(bs_data.contents[0].contents[i].contents[9].contents[3].contents[0])*scale
                        xmax = int(bs_data.contents[0].contents[i].contents[9].contents[5].contents[0])*scale
                        ymax = int(bs_data.contents[0].contents[i].contents[9].contents[7].contents[0])*scale
                    except:
                        break

                    width = (xmax - xmin)/imagex
                    height = (ymax - ymin)/imagey
                    xcenter = (xmin + width/2)/imagex
                    ycenter = (ymin + height/2)/imagey
                    classnum = 1 if onlybird else classlist.index(className)
                    textfile.write(f"{classnum} {xcenter} {ycenter} {width} {height}")
                    textfile.write('\n')



if __name__ == "__main__":
    import findClassNames

    paths = ["../quartersize_images/*.xml"]
    LableToStruc(paths, scale=1/4)

    paths = ["../quartersize_images/*.txt"]
    imagesToPath(paths)
    time.sleep(5)
    strucPath = ["../Structure/labels/train/*.xml", "../Structure/labels/test/*.xml"]
    print(findClassNames.findClasses(strucPath[0]))
    print(findClassNames.findClasses(strucPath[1]))
