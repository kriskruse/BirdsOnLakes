import os
import glob

lst3 = glob.glob("../images/*/*.xml")
print(f"Amount of XML in images: {len(lst3)}")

lst4 = glob.glob("../images2/*/*.xml")
print(f"Amount of XML in images2: {len(lst4)}")

lst5 = glob.glob("../images7/*/*.xml")
print(f"Amount of XML in images7: {len(lst5)}")

lst3 = glob.glob("../images/*/*.JPG")
print(f"Amount of JPG in images: {len(lst3)}")

lst4 = glob.glob("../images2/*/*.JPG")
print(f"Amount of JPG in images2: {len(lst4)}")

lst5 = glob.glob("../images7/*/*.JPG")
print(f"Amount of JPG in images7: {len(lst5)}")