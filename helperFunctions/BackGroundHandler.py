from MoreOfWhatWeHave import quaterSize, imageToGray, RotateImageAndData, Images


path = "backgrounds/"
savepath = "quarterBackgrounds/"
bacIm = Images(path)
N = 10

print(len(bacIm.lst))

for im in bacIm.lst:
    imagename = im.split("\\")[-1][:-4]
    bac = quaterSize(im,savepath)
    imageToGray(bac, imagename, [], savepath, "")

    for n in range(N):
        rot_bac, _ = RotateImageAndData(bac, imagename, [], savepath, "")
        imageToGray(rot_bac, imagename, [], savepath, "")
