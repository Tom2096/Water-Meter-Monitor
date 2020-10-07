#!coding=utf-8
import os
import cv2
import numpy as np
import time
from IPython import embed
from matplotlib import pyplot as plt

class Camera(object):
    def __init__(self, idx):
        self.video = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
      
    def __del__(self):
        self.video.release()
    
    def getFrame(self):
        ret, frame = self.video.read()
        if ((not ret) or frame.sum()==0): 
            return None
        return frame

def initConfig(conf):
    cxy = np.vstack([conf['center'], conf['x-axis'], conf['y-axis']]).astype('f4')
    vec_xy = cxy[1:] - cxy[0]
    len_xy = np.linalg.norm(vec_xy, axis=1) 
    unit_xy = vec_xy / len_xy
    radius = len_xy.mean()
    
    conf['radius'] = radius
    conf['unit_xy'] = unit_xy
    conf['vec_xy'] = vec_xy

def setBoundingBox(frame, conf):
    top_left = (conf['center'] - conf['radius']).astype('i4')
    bottom_right = (conf['center'] + conf['radius']).astype('i4')
    
    bbx = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return bbx

def filterAndDenoise(bbx):
    hsv_bbx = cv2.cvtColor(bbx, cv2.COLOR_BGR2HSV)

    lower_green = np.array([70, 50, 50])                                    
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv_bbx, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.erode(green_mask, kernel, iterations=1)
    green_mask = cv2.dilate(green_mask, kernel, iterations=3)
    green_mask = cv2.erode(green_mask, kernel, iterations=2)
    
    return green_mask

def findPrincipalAxis(mask):
    cords = mask.nonzero()[::-1]
    cords = np.vstack(cords)
    middle = cords.mean(axis=1).reshape(2, 1)
    cords = cords-middle

    cov = np.cov(cords)
    ei_val, ei_vec = np.linalg.eig(cov)
    ei_val = ei_val.real
    ei_vec = ei_vec.real
    
    idx = ei_val.argmax()
    axis = ei_vec[:, idx]
    projections = cords.T.dot(axis)
    
    return axis if (np.median(projections) < 0) else -axis
    
def readValue(axis, conf):

    proj_x, proj_y = conf['unit_xy'].dot(axis)
    theta = (np.arctan2(proj_y, proj_x))/(2 * np.pi)

    if proj_x > 0: return 1/4 - theta
    if proj_y < 0: return 3/4 - theta
    return 1/2 + theta

if __name__=='__main__':
    conf = {
        'cam_id':   0,
        'center': (435, 281),
        'x-axis': (460, 369),
        'y-axis': (523, 261)
    }
    
    print('[-] init camera')

    cam = Camera(conf['cam_id'])
    
    while True:
        frame = cam.getFrame()
        if frame is None:
            time.sleep(1)
            print('[-] ... retry camera')
            continue


        assert not frame is None
        print('[-] got an image')
        break

    initConfig(conf)
    bbx = setBoundingBox(frame, conf)
    mask = filterAndDenoise(bbx)
    axis = findPrincipalAxis(mask)
    res = readValue(axis, conf)
    embed()
