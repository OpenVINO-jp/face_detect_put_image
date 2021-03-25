import iewrap

import time

import cv2
import numpy as np
import image_compose as img_cmp

imgBuf = {}

def callback(infId, output):
    global imgBuf,im_back

    output = output.reshape((200,7))
    img = imgBuf.pop(infId)
    img_h, img_w, _ = img.shape

    im_figure = cv2.imread(r'rsc/foo.png')
    
    for obj in output:
        imgid, clsid, confidence, x1, y1, x2, y2 = obj
        if confidence>0.3:              # Draw a bounding box and label when confidence>0.8
            x1 = int(x1 * img_w)
            y1 = int(y1 * img_h)
            x2 = int(x2 * img_w)
            y2 = int(y2 * img_h)

            img = img_cmp.fncImgCompose(im_figure, img, (x2-x1+45,y2-y1+70),(x1-25,y1-50))

    cv2.imshow('result',img)
    cv2.waitKey(1)


def main():
    global imgBuf,im_back

    cap = cv2.VideoCapture(0)
#   cap = cv2.VideoCapture(r'.\rsc\mov\UseCase3.mp4')
    
    ie = iewrap.ieWrapper(r'.\intel\face-detection-adas-0001\FP16\face-detection-adas-0001.xml', 'CPU', 10)
    ie.setCallback(callback)

    while True:
        ret, img = cap.read()
        if ret==False:
            break

        img = cv2.resize(img,(1600,900))

        refId = ie.asyncInfer(img)     # Inference
        imgBuf[refId]=img

if __name__ == '__main__':
    main()
