import os, shutil
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm


for dir_number in tqdm(range(1,35), desc="Cropping images"):
    # control parameters
    dirc = f"images2/{dir_number}"
    folder = os.listdir(dirc)
    # print(" ")
    try:
        os.makedirs(f"images_crop2/{dir_number}")
        print(f"Created folder 'images_crop/{dir_number}'")
    except:
        print(f"Working in folder 'images_crop/{dir_number}'")

    # go over all files in the folder
    for file in folder:
        # check if it is an image
        if file.endswith(".JPG"):
            # open the image to memory
            # print(f"Found image file with name:  {file}")
            img = Image.open(f"{dirc}/{file}")
            # get the filename corresponding to the xml file that might belong to the image
            xml_filename = file.replace(".JPG", ".xml")
            # try to open the xml file, and copy it to the folder we are gonna save the images to
            try:
                xml_Data = open(f"{dirc}/{xml_filename}")
                shutil.copy(f"{dirc}/{xml_filename}", f"images_crop/{dir_number}/{xml_filename}")
                BS_Data = BeautifulSoup(xml_Data.read(), "xml")
                # print(f"Found xml file for image:  {xml_filename}")
            except FileNotFoundError:
                # print(f"No Xml file for image: {xml_filename}")
                # print("Skipping")
                continue

            # We find all the bounding boxes and loop over them
            xml_Data_object = BS_Data.find_all("bndbox")
            sets = []
            for element in xml_Data_object:
                # Getting the bounding box around the marked target
                left = int(element.contents[1].contents[0])
                top = int(element.contents[3].contents[0])
                right = int(element.contents[5].contents[0])
                bottom = int(element.contents[7].contents[0])
                sets.append((left, top, right, bottom))

            counter = 0
            for set in sets:
                # crop image and save to new folder
                img_crop = img.crop(set)
                filename = file.replace(".JPG", "")
                img_crop.save(fp=f"images_crop/{dir_number}/{filename}_{counter}.JPG")
                print(f"Saving cropped image to:  images_crop/{dir_number}/{counter}_{file}")
                counter += 1

            # stopping memory leaking
            img.close()
            xml_Data.close()
