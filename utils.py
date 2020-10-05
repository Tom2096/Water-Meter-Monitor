#!coding=utf-8

import os
import cv2
import numpy as np


def _eigen(arr):             # arr:  nx2
    mu = np.mean(arr,axis=0)    # mu:   nx2 -> 2,
    evals, evcts = np.linalg.eig( np.cov((arr-mu).T) )  # evals: 2 , evcts: 2x2
    evals, evcts = evals.real, evcts.real  
        
    eInd    = (np.argsort(evals))[::-1]   
    return evals[eInd], evcts.T[eInd], mu      


def _calculate_info(conf):
    cyx = np.vstack([conf['center'], conf['y-axis'], conf['x-axis']]).astype('f4')
    yx = cyx[1:] - cyx[0]
    len_yx = np.linalg.norm(yx, axis=1)
    axis = yx/len_yx
    return axis, len_yx.mean()


def _clip_bbox(conf):
    p0 = conf['center'] - conf['_radius']*1.05
    p1 = conf['center'] + conf['_radius']*1.05
    return np.vstack([p0,p1]).astype('i4')


def _filter_by_color(hsv):
    #-- lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    #-- upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    return mask0+mask1
    

def _find_main_axis(mask):
    #-- find out main axis 
    arr = mask.nonzero()            # y,x
    arr = np.c_[arr[::-1]]          # x,y
    vals, vcts, mu = _eigen(arr)
    
    #-- adjust direction
    line = vcts[0]
    h,w = mask.shape

    _arr = arr - np.r_[h,w]/2
    _v = _arr.dot(line)
    if np.median(_v)<0:
        line = -line
    
    return line

def _calculate_theta(line, conf):
    project_y, project_x = conf['_yx'].dot(line)
    theta = np.arctan2(project_y, project_x)        # [-pi, pi]

    if project_x>0.01 or project_y<0:
        theta = np.pi/2 - theta
    elif project_y>0 and project_x<-0.01:
        theta = 2*np.pi - (theta-np.pi/2)
    else:
        theta = None    # too close  +y-axis, not stable
    
    return theta



#-------------------------------------------
def prepare_conf(conf):
    yx, radius = _calculate_info(conf)
    conf['_yx'] = yx
    conf['_radius'] = radius

    bbx = _clip_bbox(conf)
    conf['_bbx'] = bbx



def extract_value(im, conf):
    #-- clip image, converte to hsv color-space
    bbx = conf['_bbx']
    clip_im = im[bbx[0,1]:bbx[1,1], bbx[0,0]:bbx[1,0]]
    hsv =cv2.cvtColor(clip_im, cv2.COLOR_BGR2HSV)
    
    mask = _filter_by_color(hsv)
    
    #-- de-noise    
    kernel = np.ones((3,3),np.uint8)  
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 3)
    mask = cv2.erode(mask,kernel,iterations = 2)

    line = _find_main_axis(mask)
    theta = _calculate_theta(line, conf)
    
    #---        
    if theta is None:   return None

    return theta/(2*np.pi)



class VideoCamera(object):

    def __init__(self, idx):
        self.video = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            
    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, buf = self.video.read()
        if not ret or buf is None or buf.sum()==0:
            return None
        
        return buf



if __name__=='__main__':
    conf = {
        'cam_id':   1,
        
        'center':   (435,281),      # <px,py>
        'y-axis':   (523,261),
        'x-axis':   (460,369),
    }
    
    prepare_conf(conf)
    print('[-] init camera')
    cam = VideoCamera(conf['cam_id'])
    
    while True:
        data = cam.get_frame()
        if data is None:
            time.sleep(1)
            print('[-] ... retry camera')
            continue

        assert not data is None
        print('[-] got an image')
        
        #-- process 
        val = extract_value(data, conf)
        print('[-] val is %s'%val)
        break

    print('[-] ---- over ----')

