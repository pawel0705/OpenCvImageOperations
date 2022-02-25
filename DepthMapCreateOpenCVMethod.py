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


# Zadanie 3


# In[11]:


img_left = cv.imread('egzamin0/t0z3a/im0.png', 0)
img_right = cv.imread('egzamin0/t0z3a/im1.png', 0)
num_disparities = 138
block_size = 25

stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disparities,
    blockSize=block_size,
    speckleWindowSize=40,
    speckleRange=17,
    preFilterCap=10
)

stereo = stereo / 16 # ?????

disparity = stereo.compute(img_left, img_right)

plt.imshow(disparity / 256, 'gray')
plt.savefig('disparityZad3.png', dpi = 300)
plt.show()


# In[12]:


def calculateDisparityToDepth(disparity, baseline, f):
    Z = (baseline * f)/disparity
    return Z


# In[13]:


depth = calculateDisparityToDepth(disparity, 0.53662, 1758.23)


# In[15]:


plt.imshow(depth, 'gray')
plt.savefig('depthZad3.png', dpi = 300)
plt.show()


# In[7]:


groundtruth = cv.imread('egzamin0/t0z3a/disp0.pfm', cv.IMREAD_UNCHANGED)
groundtruth = np.asarray(groundtruth)
groundtruth = groundtruth / 256

plt.imshow(groundtruth, 'gray')
plt.show()


# In[8]:


groundtruthDepth = calculateDisparityToDepth(groundtruth, 0.53662, 1758.23)

plt.imshow(groundtruthDepth, 'gray')
plt.show()


# In[9]:


print("point depth_gt", groundtruthDepth[755,588])
print("point depth", depth[755,588])
print("point disparity_gt", groundtruth[755,588])
print("point disparity", disparity[755,588])


# In[10]:


cv.imwrite('disparity.png',disparity)
cv.imwrite('depth.png',depth)


# In[ ]:


depht_range = 256.0 * 256.0 * 256.0 - 1
normalizer = 255
d = depth * depht_range;

b = d.astype(int) >> 16
g = d.astype(int) >> 8
r = d.astype(int)

bit24depth = np.zeros([depth.shape[0],depth.shape[1],3],dtype=np.uint8)


bit24depth[..., 0] = b / normalizer
bit24depth[..., 1] = g / normalizer
bit24depth[..., 2] = r / normalizer

plt.imshow(bit24depth[:,:,::-1])

