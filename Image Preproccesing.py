import os
from xml.dom import minidom
from PIL import Image

folder = os.listdir("images/something")

for file in folder:
    if file.endswith(".JPG"):
        img = Image.open(file)
        xml_filename = file.replace(".JPG", ".xml")
        try:
            xml_data = minidom.parse(xml_filename)
        except:
            continue
        left = data[""]
        top = data[""]
        right = data[""]
        bottom = data[""]

        img_crop = img.crop((left, top, right, bottom))
        img_crop.save(fp = f"images_crop/{file}")