import os
import glob
import numpy as np
import Iterator
#This one needs to be installed!
import imageio
import cv2

def make_giff( it, cvfun, filename, use_kernel = False):
    
    images = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    while(1): 
        try:
            im, gt = next(y)
            fgmask = cvfun.apply(im.astype(np.uint8))
            if use_kernel:
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            #and write your frames in a loop if you want
            images.append(fgmask)

        except StopIteration:
            break
    imageio.mimsave(filename, images)
    
    return
