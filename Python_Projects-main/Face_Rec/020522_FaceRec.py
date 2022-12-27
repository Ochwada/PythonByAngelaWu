#from calendar import c
import imp


import cv2
from matplotlib.cbook import ls_mapper
import numpy as np
import time
import os

# function to get images from folder 
def get_images(dir_name):
    list_images = os.listdir(dir_name)
    all_images =list()

    for entry in list_images:
        full_path =os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            all_images.all_images + get_images(full_path)
        else: all_images.append(full_path)
    return all_images

# Face Recognition

def  main():
    dir_name = 'images' # directory for images
    list_images = get_images(dir_name)

    for i in range(20): #20 images
        image_path = list_images[i]

        print(image_path)
        
        # load the pre-trained model
        case_path = "haarcascade_frontalface_default.xml" # define the model used for recognition detection
        faceCascade= cv2.CascadeClassifier(case_path)
        image=cv2.imread(image_path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # make the picture to gray from color

        # face detection 
        faces = faceCascade.detectMultiScale(gray,
                                            scaleFactor = 1.1,
                                            minNeighbors=5,
                                            minSize = (30, 30))

        for (x, y, w, h) in faces: # draw rectangles on the faces when detected
            cv2.rectangle(image, (x,y), (x + w, y+h), (0, 255, 0), 2)

        #Load the detected faces
        cv2.imshow("Face Found", image)
        cv2.waitKey(4)
        time.sleep(5)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

## for Beginners
