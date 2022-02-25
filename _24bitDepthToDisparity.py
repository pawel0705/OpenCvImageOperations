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


# Zadanie 2


# In[3]:


def calculateCarlaAsDeep(carla):
    carla = carla[:, :, :3]  
    carla = carla[:, :, ::-1] 
    depth = ((carla[:, :, 0] + carla[:, :, 1] * 256.0 + carla[:, :, 2]*256.0*256.0)/((256.0*256.0*256.0) - 1))
    depth = depth * 1000
    return depth


# In[4]:


def calculateDepthToDisparity(depthMap, baseline, fov):
    h, w = depthMap.shape[:2]
    focusL = w / (2 * math.tan(fov * math.pi/360))
    disparityMap = (baseline*focusL)/depthMap
    return disparityMap


# In[5]:


carla_img = cv.imread('egzamin0/t0z1a/depth.png')

depth = calculateCarlaAsDeep(carla_img)

baseline = 0.6
FOV = 120

plt.imshow(depth, 'gray')
plt.show()


# In[6]:


disparity = calculateDepthToDisparity(depth, baseline, FOV)
disparity = disparity.astype(np.uint8)
plt.imshow(disparity, 'gray')
plt.show()


# In[7]:


print("point", disparity[400,1600])


# In[8]:


cv.imwrite('zad2_disparity_egzamin.png',disparity)


# In[ ]:




