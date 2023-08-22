import cv2 as cv
import numpy as np

PTMAP = [
    [7,  0, 1],
    [6,  8, 2],
    [5,  4, 3]
]
NEW_SIZE = (60,40)

def preprocess(img, fname):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=255)
    _, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)

    return img

def incrementContourHistogram(contour, hist):
    prevpt = contour[0][0]
    for i in range(1,len(contour)):
        curpt = contour[i][0]

        map_i = curpt[0]-prevpt[0]
        map_j = curpt[1]-prevpt[1]
        if map_i != 0: map_i = map_i // abs(map_i)
        if map_j != 0: map_j = map_j // abs(map_j)
        map_i += 1; map_j += 1 # map i and j are coordinates to the PTMAP matrix according to contour direction

        hist[ PTMAP[map_i][map_j] ] += 1

        prevpt = curpt
    
    return hist

def getImgSections(img, v_divs, h_divs):
    hei, wid = img.shape
    subimg_hei = hei // v_divs
    subimg_wid = wid // h_divs

    # Separate image into sections
    imgSections = []
    for i in range(v_divs):
        for j in range(h_divs):
            up = i * subimg_hei
            down = (i+1) * subimg_hei
            left = j * subimg_wid
            right = (j+1) * subimg_wid
            imgSections.append(img[up:down, left:right])
    return imgSections

def getImgCharacteristics(img):
    imgSections = getImgSections(img, 3, 3)

    characteristics = []
    densities = []
    for imgSection in imgSections:
        contours, hierarchy = cv.findContours(imgSection, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        hist = [0]*8
        for contour in contours:
            hist = incrementContourHistogram(contour, hist)
        characteristics.extend(hist)

        densities.append( getPixelDensity(imgSection) )
    
    characteristics = [v/max(characteristics) for v in characteristics]
    characteristics.extend(densities)

    return characteristics

def getPixelDensity(img):
    num_white = np.sum(img == 255)
    total = img.shape[0] * img.shape[1]
    return num_white/total


def showContours(contours):
    blackimg = np.zeros(NEW_SIZE, dtype=np.uint8)
    cv.drawContours(blackimg, contours, -1, 127)
    cv.imshow("contours", blackimg)
    cv.waitKey(0)
    
def outputCharacteristics(characteristics, label):
    print(label, end=" ")
    for ch in characteristics:
        print(ch, end=" ")
    print()

DATASET_DIR = "digits/"
if __name__ == '__main__':

    f = open(DATASET_DIR + "files.txt", "r")
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        imgFile = DATASET_DIR + line[0]
        label = line[1]

        cvImg = cv.imread(imgFile)
        assert cvImg is not None, "file could not be read, check with os.path.exists()"

        
        cvImg = preprocess(cvImg, line[0])
        characteristics = getImgCharacteristics(cvImg)
        outputCharacteristics(characteristics, label)
    