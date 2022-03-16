import cv2

img_bird = cv2.imread("images/10/IMAG0032.JPG")
img_back = cv2.imread("images/10/IMAG0050.JPG")
img_bird = cv2.cvtColor(img_bird, cv2.COLOR_BGR2GRAY)
img_back = cv2.cvtColor(img_back, cv2.COLOR_BGR2GRAY)

no_back = img_bird - img_back
cv2.imwrite("no_background.JPG", no_back)
cv2.imwrite("gray_bird.JPG", img_bird)
cv2.imwrite("gray_back.JPG", img_back)

