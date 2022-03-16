import os
import glob

lst = glob.glob("images_crop/*/*.JPG")
lst2 = glob.glob("images_crop/*/*.XML")
print(f"Amount of birds in images_crop: {len(lst)}")
print(f"Amount of XML in images_crop: {len(lst2)}")