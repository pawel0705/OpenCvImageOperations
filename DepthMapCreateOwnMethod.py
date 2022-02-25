#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv
import glob
import os
import json
from json import JSONEncoder
import matplotlib.pyplot as plt
import math
import timeit
import yaml


# In[2]:


# Zadanie 4


# In[3]:


img_left = cv.imread('egzamin0/t0z3a/im0.png', 0)
img_right = cv.imread('egzamin0/t0z3a/im1.png', 0)

originalH, originalW = img_left.shape
print(originalW)
print(originalH)

disparity_map_tmp = np.zeros((originalH, originalW))

#define kernel size
kernelW = 25
kernelH = 25

for x in range(originalW):
    if x + kernelW > originalW:
        break
    for y in range(originalH):
        if y + kernelH > originalH:
            break
        
        # define kernel
        kernel = img_left[y:y+kernelW, x:x+kernelH]
        # Apply template Matching
        
        fromY = y
        if fromY > 1:
            fromY -= 1
        
        searchFrom = x - 200
        searchTo = x + 200
        
        if searchFrom < 16:
            searchFrom = 16
            
        if searchTo > originalW:
            searchTo = originalW
        
        searchImage = img_right[fromY:fromY+kernelH, searchFrom:searchTo]
        res = cv.matchTemplate(searchImage,kernel,cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        disparity_map_tmp[y, x] = max_loc[0] + searchFrom


# In[18]:


disparity_map = np.zeros((originalH, originalW))

for x in range(originalW):
    for y in range(originalH):
        if x + kernelW > originalW:
            continue
        if y + kernelH > originalH:
            continue
        
        disp = x - disparity_map_tmp[y][x];

        
        if(disp < 0):
            disp = 0
            
        disparity_map[y][x] = disp;
        
plt.imshow(disparity_map, cmap='gray')
plt.savefig('disparityZad4.png', dpi = 300)
plt.show()


# In[5]:


def calculateDisparityToDepth(disparity, baseline, f):
    h, w = disparity.shape[:2]
    #focusL = w / (2 * math.tan(fov * math.pi / 360))
    Z = (baseline * f)/disparity
    return Z


# In[7]:


depth = calculateDisparityToDepth(disparity_map, 0.53662, 1758.23)


# In[19]:


plt.imshow(depth, 'gray')
plt.savefig('depthZad4.png', dpi = 300)
plt.show()


# In[9]:


groundtruth = cv.imread('egzamin0/t0z3a/disp0.pfm', cv.IMREAD_UNCHANGED)
groundtruth = np.asarray(groundtruth)
groundtruth = groundtruth / 256

plt.imshow(groundtruth, 'gray')
plt.show()


# In[11]:


groundtruthDepth = calculateDisparityToDepth(groundtruth, 0.53662, 1758.23)

plt.imshow(groundtruthDepth, 'gray')
plt.show()


# In[13]:


print("point depth_gt", groundtruthDepth[755,588])
print("point depth", depth[755,588])
print("point disparity_gt", groundtruth[755,588])
print("point disparity", disparity_map[755,588])


# In[15]:


cv.imwrite('disparity2.png',disparity_map)
cv.imwrite('depth2.png',depth)


# In[ ]:




