import cv2 as cv
import numpy as np

PTMAP = [
    [7,  0, 1],
    [6,  8, 2],
    [5,  4, 3]
]

def incrementContourHistogram(contour, hist):
    prevpt = contour[0][0]
    for i in range(1,len(contour)):
        curpt = contour[i][0]

        map_i = curpt[0]-prevpt[0]
        map_j = curpt[1]-prevpt[1]
        if map_i != 0: map_i = map_i // abs(map_i)
        if map_j != 0: map_j = map_j // abs(map_j)
        map_i += 1; map_j += 1

        hist[ PTMAP[map_i][map_j] ] += 1

        prevpt = curpt
    
    return hist

def preprocess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, NEW_SIZE)
    _, thresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)
    return img

def getCharacteristics(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    showContours(contours)

    hist = [0]*8
    for contour in contours:
        hist = incrementContourHistogram(contour, hist)
    return hist

def showContours(contours):
    blackimg = np.zeros(NEW_SIZE, dtype=np.uint8)
    cv.drawContours(blackimg, contours, -1, 127)
    cv.imshow("contours", blackimg)
    cv.waitKey(0)
    
    

DATA_DIR = "digits/data/"
NEW_SIZE = (50,50)
if __name__ == '__main__':

    imgFile = DATA_DIR + "cdf0361_07_13_0.jpg"
    cvImg = cv.imread(imgFile)
    assert cvImg is not None, "file could not be read, check with os.path.exists()"

    # preprocess image
    cvImg = preprocess(cvImg)
    characteristics = getCharacteristics(cvImg)

    print(characteristics)