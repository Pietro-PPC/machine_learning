import cv2 as cv
import numpy as np

DATA_DIR = "digits/data/"

imgFile = DATA_DIR + "cdf0361_07_13_0.jpg"

cvImg = cv.imread(imgFile)
assert cvImg is not None, "file could not be read, check with os.path.exists()"

cvImg = cv.cvtColor(cvImg, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(cvImg, 127, 255, cv.THRESH_BINARY_INV)


contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(thresh, contours, -1, 128)

print(contours[0])

# print(thresh)
cv.imshow("img", thresh)
cv.waitKey(0)
