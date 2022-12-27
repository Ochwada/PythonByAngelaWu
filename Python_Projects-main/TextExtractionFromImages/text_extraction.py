

import cv2
import numpy as np
import pytesseract as pyt


image = cv2.imread('The_Zen_of_Python.jpeg')

# image convertion to grey scale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray, binary_image = cv2.threshold(image_gray, 128,255, cv2.THRESH_BINARY)
image_gray =cv2.bitwise_not(binary_image)

kernel = np.ones((2,1), np.uint8)
image = cv2.erode(image_gray, kernel,iterations = 1)
image = cv2.dilate(image_gray, kernel,iterations = 1)

# image convertion into string

image_text = pyt.image_to_string(image)

print(image_text)