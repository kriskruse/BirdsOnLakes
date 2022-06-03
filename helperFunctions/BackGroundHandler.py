from MoreOfWhatWeHave import quaterSize, imageToGray, RotateImageAndData, Images


path = "Test_backgrounds_originals/"
savepath = "Test_quaterbackgrounds/"
bacIm = Images(path)
N = 3

print(len(bacIm.lst))

for im in bacIm.lst:
    imagename = im.split("\\")[-1][:-4]
    bac = quaterSize(im,savepath)

    # Uncomment this for training backgrounds
    # imageToGray(bac, imagename, [], savepath, "")
    #
    # for n in range(N):
    #     rot_bac, _ = RotateImageAndData(bac, imagename, [], savepath, "")
    #     imageToGray(rot_bac, imagename, [], savepath, "")
