from MoreOfWhatWeHave import Images, quaterSize, xmlToTxt
import os

path = "../images7/*"
savepath = "OneClassTestSet/images"
labelsavepath = "OneClassTestSet/labels"
classlist = None
onlybird = True

img = Images(path)
os.makedirs(savepath, exist_ok=True)
os.makedirs(labelsavepath, exist_ok=True)

for i in range(img.size):
    imagepath = img.lst[i]
    xmlpath = img.xml_lst[i]
    imagename = xmlpath.split("\\")[-1][:-4]
    image = quaterSize(imagepath, savepath)
    objectdata = xmlToTxt(xmlpath, labelsavepath, imagename, classlist, scale=1 / 4, onlybird=onlybird)
