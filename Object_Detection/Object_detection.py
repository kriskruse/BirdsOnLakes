
# Notes
# Removing background? is it a problem that the background is there? we might be able to remove it if we subtract the
# image we are looking at with a background only image. This background only image could be a mean of different
# background images since our background changes alot

# If we use a premade model, can we train it to only detect birds? or can we clean the output to only focus on the birds
# Can we make the model also classify the bird species or only whether there is a bird.

# A model suggestion is the YOLO model, currently "YOLO V5", however this model is heavy pretrained.
#

from imageai.Detection import ObjectDetection

detector = ObjectDetection()

model_path = "./models/yolo-tiny.h5"
input_path = "./input/IMAG0535.JPG"
output_path = "./output/newimage.jpg"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()

detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])