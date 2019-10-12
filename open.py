import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
fgbg = cv.bgsegm.createBackgroundSubtractorMOG(history=5)
while(1):
    ret, frame = cap.read()
    imgray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # ret, thresh = cv.threshold(imgray, 127, 255, 0)
    thresh = fgbg.apply(imgray)
    motionCounter = sum(sum(thresh))
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                               cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # fgmask = fgbg.apply(frame)
    # cv.imshow('frame',np.stack(np.multiply(fgmask,frame[:,:,0]), np.multiply(fgmask,frame[:,:,1]),np.multiply(fgmask,frame[:,:,2])))
    cv.imshow('frame',frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()


# fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
# while(1):
#     ret, frame = cap.read()
#     fgmask = fgbg.apply(frame)
#     cv.imshow('frame',np.multiply(fgmask,frame[:,:,1]))
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
# cap.release()
# cv.destroyAllWindows()

# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
# fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
# while(1):
#     ret, frame = cap.read()
#     fgmask = fgbg.apply(frame)
#     fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
#     cv.imshow('frame',fgmask)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
# cap.release()
# cv.destroyAllWindows()