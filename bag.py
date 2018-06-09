import cv2
import sys
import numpy as np

from matplotlib import pyplot as plt

name = sys.argv[1]
op = sys.argv[2]
show = sys.argv[3]
if(len(sys.argv)>4):
    t = int(sys.argv[4])
else:
    t = 40

def bgr2rgb(img):
    b, g, r = cv2.split(img)
    r = r.astype(np.float)
    g = g.astype(np.float)
    b = b.astype(np.float)
    return r, g, b

def hsv(img):
    h, s, v  = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    ret, dst = cv2.threshold(s*(v/255.0),t,255,cv2.THRESH_BINARY)
    return dst

def mhsv(r, g, b):
    img = np.uint8(np.sqrt((g-b)**2 + (b-r)**2 + (r-g)**2)/np.sqrt(2))
    ret, dst = cv2.threshold(img,t,255,cv2.THRESH_BINARY)
    return dst

def showplt(img):
    plt.imshow(img, cmap='gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def showim(img):
    cv2.imshow("bag",img)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()

img = cv2.imread(name)
if op == 'd':
    res = hsv(img)
elif op == 'm':
    r, g, b = bgr2rgb(img)
    res = mhsv(r, g, b)
else:
    exit()
if 1:
    kernel = np.ones((5,5))
    res = cv2.erode(res,kernel)
    h, s, v  = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    res[res!=0] = h[res!=0]
if show == 'p':
    showplt(res)
if show == 'i':
    showim(res)