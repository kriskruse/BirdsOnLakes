import cv2
import numpy as np
import glob
import os

# Get the size we want to achieve, uses an image so we can match that. can be set manually
target_image = cv2.imread('images/1/IMAG0030.JPG')
new_image_height, new_image_width, _ = target_image.shape
color = (0, 0, 0)

folder_pat = "Kaggle_MOdel/train/*/*"

for image_path in glob.glob(folder_pat):
    # read image
    img = cv2.imread(image_path)
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color (black) for padding
    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center + old_image_height,
    x_center:x_center + old_image_width] = img

    _ , new_image_path = image_path.split("/")
    # save result
    dir_path = f"Kaggle_MOdel/padded/{new_image_path}"
    os.makedirs(dir_path[0:-8] , exist_ok=True)

    cv2.imwrite(f"Kaggle_MOdel/padded/{new_image_path}", result)
    print(f"{image_path}: Padded and saved to: Kaggle_MOdel/padded/{new_image_path}")
