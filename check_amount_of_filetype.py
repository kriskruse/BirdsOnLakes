import os
import glob

lst = glob.glob("images_crop/*/*.JPG")
lst2 = glob.glob("images_crop/*/*.XML")
print(f"Amount of birds in images_crop: {len(lst)}")
print(f"Amount of XML in images_crop: {len(lst2)}")

lst3 = glob.glob("images/*/*.XML")
print(f"Amount of XML in images: {len(lst3)}")

lst4 = glob.glob("images2/*/*.XML")
print(f"Amount of XML in images2: {len(lst4)}")