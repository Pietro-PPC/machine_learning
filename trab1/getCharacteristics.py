import cv2 as cv
import numpy as np

PTMAP = [
    [7,  0, 1],
    [6, -1, 2],
    [5,  4, 3]
]

def incrementContourHistogram(contour, hist):
    prevpt = contour[0][0]
    for i in range(1,len(contour)):
        curpt = contour[i][0]

        map_i = curpt[0]-prevpt[0]
        map_j = curpt[1]-prevpt[1]
        map_i = map_i // abs(map_i) + 1
        map_j = map_j // abs(map_j) + 1

        hist[ PTMAP[map_i][map_j] ] += 1

        prevpt = curpt
    
    return hist

DATA_DIR = "digits/data/"
NEW_SIZE = (50,50)

imgFile = DATA_DIR + "cdf0361_07_13_0.jpg"
cvImg = cv.imread(imgFile)
assert cvImg is not None, "file could not be read, check with os.path.exists()"

cvImg = cv.cvtColor(cvImg, cv.COLOR_BGR2GRAY)
cvImg = cv.resize(cvImg, NEW_SIZE)
_, thresh = cv.threshold(cvImg, 200, 255, cv.THRESH_BINARY_INV)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

hist = [0]*8
for contour in contours:
    incrementContourHistogram(contour, hist)

blackimg = np.zeros(NEW_SIZE, dtype=np.uint8)
cv.drawContours(blackimg, contours, -1, 128)

print(hist)
cv.imshow("img", blackimg)
cv.waitKey(0)
