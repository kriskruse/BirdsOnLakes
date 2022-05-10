from glob import glob
from os import listdir
from PIL import Image

broken = []
for filename in listdir('./oneClassDataSet_proper/images/'):
    if filename.endswith('.jpg'):
        try:
            img = Image.open('./oneClassDataSet_proper/images/' + filename)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)  # print out the names of corrupt files
            broken.append(filename)

print(broken)
print(len(broken))