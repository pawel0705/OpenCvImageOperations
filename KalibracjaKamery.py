# Kalibrowanie kamery na podstawie sekwencji obrazów z szachownicą.

import numpy as np
import cv2 as cv
import glob
import os
import json
from json import JSONEncoder
import matplotlib.pyplot as plt


# In[2]:


# Wykrywanie wzorca kalibracyjnego

calibration_pattern_dimension = 35.00 #[mm] 35 mm
calibration_pattern_width = 8
calibration_pattern_height = 6

#Zbiór k0

left_images = []
right_images = []
left_right_images = []

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

imagesS1 = glob.glob('k0/*.png')

# znajdywanie szachownicy na obrazkach
for fname in imagesS1:
    print(fname)
    try:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (calibration_pattern_width,calibration_pattern_height), None)
        # if found
        if ret == True:
            if "left" in fname:
                left_images.append(fname)
            elif "right" in fname:
                right_images.append(fname)
    except Exception as e:
        print("Excepion for image: " + fname)
        print(e)


# In[3]:


# zapis obrazków na których znaleziona została szachownica
with open("left_images_k0.txt", "w") as f:
    for s in left_images:
        f.write(str(s) +"\n")

        
with open("right_images_k0.txt", "w") as f:
    for s in right_images:
        f.write(str(s) +"\n")


# In[4]:


# zapis par obrazków
for left in left_images:
    for right in right_images:
        left_tmp = left.replace('left','')
        right_tmp = right.replace('right','')
        if(left_tmp == right_tmp):
            print(left + '-' + right)
            left_right_images.append(left)
            left_right_images.append(right)

with open("left_right_images_k0.txt", "w") as f:
    for s in left_right_images:
        f.write(str(s) +"\n")


# In[5]:


# odczyt lewych i prawych obrazków po nazwach z pliku, gdzie znaleziono szachownicę
read_left_images = []
read_right_images = []

with open("left_images_k0.txt", "r") as f:
    for line in f:
        read_left_images.append(line.strip())
        
with open("right_images_k0.txt", "r") as f:
    for line in f:
        read_right_images.append(line.strip())

print("Liczba lewych obrazków z szachownicą:")
print(len(read_left_images))
print("Liczba prawych obrazków z szachownicą:")
print(len(read_right_images))


# In[6]:


#  Wyznaczanie parametrów macierzy wewnętrznej

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp_left = np.zeros((calibration_pattern_height*calibration_pattern_width,3), np.float32)
objp_left[:,:2] = np.mgrid[0:calibration_pattern_width,0:calibration_pattern_height].T.reshape(-1,2)

objp_left *= calibration_pattern_dimension

objp_right = np.zeros((calibration_pattern_height*calibration_pattern_width,3), np.float32)
objp_right[:,:2] = np.mgrid[0:calibration_pattern_width,0:calibration_pattern_height].T.reshape(-1,2)

objp_right *= calibration_pattern_dimension

# Arrays to store object points and image points from all the images.
objpoints_left = [] # 3d point in real world space
imgpoints_left = [] # 2d points in image plane.

objpoints_right = [] # 3d point in real world space
imgpoints_right = [] # 2d points in image plane.


# In[7]:


max_images = 40
images_iterator = 0

# Kalibracja dla lewej kamery
for fname in read_left_images:
    try:
        if images_iterator > max_images:
            break
        
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (calibration_pattern_width,calibration_pattern_height), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints_left.append(objp_left)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints_left.append(corners2)
            images_iterator += 1
            # Draw and display the corners
            #cv.drawChessboardCorners(img, (calibration_pattern_width,calibration_pattern_height), corners2, ret)
            #cv.imshow('img', img)
            #cv.waitKey(1)
    except Exception as e:
        print("Excepion for image: " + fname)
        print(e)

cv.destroyAllWindows()


# In[8]:


ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv.calibrateCamera(objpoints_left, imgpoints_left, gray.shape[::-1], None, None)


# In[9]:


# Wyświetl wyniki kalibracji dla lewej kamery
print(ret_left)
print(mtx_left)
print(dist_left)
print(rvecs_left)
print(tvecs_left)


# In[10]:


max_images = 40
images_iterator = 0

# Kalibracja dla prawej kamery
for fname in read_right_images:
    try:
        if images_iterator > max_images:
            break
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (calibration_pattern_width,calibration_pattern_height), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints_right.append(objp_right)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints_right.append(corners2)
            images_iterator += 1
            # Draw and display the corners
            #cv.drawChessboardCorners(img, (calibration_pattern_width,calibration_pattern_height), corners2, ret)
            #cv.imshow('img', img)
            #cv.waitKey(1)
    except Exception as e:
        print("Excepion for image: " + fname)
        print(e)

cv.destroyAllWindows()


# In[11]:


ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv.calibrateCamera(objpoints_right, imgpoints_right, gray.shape[::-1], None, None)


# In[12]:


# Wyświetl wyniki kalibracji dla prawej kamery

print(ret_right)
print(mtx_right)
print(dist_right)
print(rvecs_right)
print(tvecs_right)


# In[13]:


# Średni błąd reprojekcji

# lewa kamera
mean_error_left = 0
for i in range(len(objpoints_left)):
    imgpoints2, _ = cv.projectPoints(objpoints_left[i], rvecs_left[i], tvecs_left[i], mtx_left, dist_left)
    error = cv.norm(imgpoints_left[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error_left += error
    
print("Mean reprojection error LEFT camera: ", mean_error_left/len(objpoints_left))

# prawa kamera
mean_error_right = 0
for i in range(len(objpoints_right)):
    imgpoints2, _ = cv.projectPoints(objpoints_right[i], rvecs_right[i], tvecs_right[i], mtx_right, dist_right)
    error = cv.norm(imgpoints_right[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error_right += error
    
print("Mean reprojection error RIGHT camera: ", mean_error_left/len(objpoints_right))


# In[14]:


# Zapis parametrów kalibracyjnych kamery do formatu JSON
# zapisz uzyskane wyniki do jsona
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# In[15]:


json_dump_left = json.dumps({'ret': ret_left, 'mtx': mtx_left , 'dist': dist_left , 'rvecs': rvecs_left, 'tvecs': tvecs_left}, cls=NumpyEncoder)
json_dump_right = json.dumps({'ret': ret_right, 'mtx': mtx_right , 'dist': dist_right , 'rvecs': rvecs_right, 'tvecs': tvecs_right}, cls=NumpyEncoder)


with open('dataLeft_k0.json', 'w') as outfile:
    json.dump(json_dump_left, outfile)
    
with open('dataRight_k0.json', 'w') as outfile:
    json.dump(json_dump_right, outfile)


# In[16]:


# Usuwanie dystorsji na obrazie - metoda undistort

# odczytaj wyniki z pliku

data_read_left = {}
data_read_right = {}

with open('dataLeft_k0.json') as json_file:
    data_read_left = json.load(json_file)

with open('dataRight_k0.json') as json_file:
    data_read_right = json.load(json_file)
    
data_read_left = json.loads(data_read_left)
data_read_right = json.loads(data_read_right)


# In[17]:


print(data_read_left)
print(data_read_right)


# In[18]:


ret_left_read = data_read_left['ret']
mtx_left_read = np.array(data_read_left['mtx'])
dist_left_read = np.array(data_read_left['dist'])
rvecs_left_read = np.array(data_read_left['rvecs'])
tvecs_left_read = np.array(data_read_left['tvecs'])

ret_right_read = data_read_right['ret']
mtx_right_read = np.array(data_read_right['mtx'])
dist_right_read = np.array(data_read_right['dist'])
rvecs_right_read = np.array(data_read_right['rvecs'])
tvecs_right_read = np.array(data_read_right['tvecs'])


# In[19]:


#test
print(ret_right_read)
print(mtx_right_read)
print(dist_right_read)
print(rvecs_right_read)
print(tvecs_right_read)


# In[20]:


#Undystorsja lewa kamera

# load image
img = cv.imread('k0/left_63.png')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx_left_read, dist_left_read, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx_left_read, dist_left_read, None, newcameramtx)
# crop the image
#x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
cv.imwrite('calibration_left_63_k0.png', dst)


#Undystorsja prawa kamera

# load image
img = cv.imread('k0/right_63.png')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx_right_read, dist_right_read, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx_right_read, dist_right_read, None, newcameramtx)
# crop the image
#x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
cv.imwrite('calibration_right_63_k0.png', dst)


# In[21]:


#stereo kalibracja

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp_left_pair = np.zeros((calibration_pattern_height*calibration_pattern_width,3), np.float32)
objp_left_pair[:,:2] = np.mgrid[0:calibration_pattern_width,0:calibration_pattern_height].T.reshape(-1,2)

objp_left_pair *= calibration_pattern_dimension

objp_right_pair = np.zeros((calibration_pattern_height*calibration_pattern_width,3), np.float32)
objp_right_pair[:,:2] = np.mgrid[0:calibration_pattern_width,0:calibration_pattern_height].T.reshape(-1,2)

objp_right_pair *= calibration_pattern_dimension

# Arrays to store object points and image points from all the images.
objpoints_left_pair = [] # 3d point in real world space
imgpoints_left_pair = [] # 2d points in image plane.

objpoints_right_pair = [] # 3d point in real world space
imgpoints_right_pair = [] # 2d points in image plane.


# In[22]:


read_pairs_images = []

with open("left_right_images_k0.txt", "r") as f:
    for line in f:
        read_pairs_images.append(line.strip())


# In[23]:


# read RIGHT images from pair
images_iterator = 0
max_images = 40
for fname in read_pairs_images:
    try:
        if "left" in fname:
            continue;
        
        if images_iterator > max_images:
            break;
        
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (calibration_pattern_width,calibration_pattern_height), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            images_iterator += 1
            objpoints_right_pair.append(objp_right_pair)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints_right_pair.append(corners2)
            # Draw and display the corners
            #cv.drawChessboardCorners(img, (calibration_pattern_width,calibration_pattern_height), corners2, ret)
            #cv.imshow('img', img)
            #cv.waitKey(1)
    except Exception as e:
        print("Excepion for image: " + fname)
        print(e)

cv.destroyAllWindows()
print(images_iterator)


# In[24]:


ret_right_pair, mtx_right_pair, dist_right_pair, rvecs_right_pair, tvecs_right_pair = cv.calibrateCamera(objpoints_right_pair, imgpoints_right_pair, gray.shape[::-1], None, None)


# In[25]:


# read LEFT images from pair
images_iterator = 0
max_images = 40
for fname in read_pairs_images:
    try:
        if "right" in fname:
            continue;
        
        if images_iterator > max_images:
            break;
        
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (calibration_pattern_width,calibration_pattern_height), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            images_iterator+=1
            objpoints_left_pair.append(objp_left_pair)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints_left_pair.append(corners2)
            # Draw and display the corners
            #cv.drawChessboardCorners(img, (calibration_pattern_width,calibration_pattern_height), corners2, ret)
            #cv.imshow('img', img)
            #cv.waitKey(1)
    except Exception as e:
        print("Excepion for image: " + fname)
        print(e)

cv.destroyAllWindows()
print(images_iterator)


# In[26]:


ret_left_pair, mtx_left_pair, dist_left_pair, rvecs_left_pair, tvecs_left_pair = cv.calibrateCamera(objpoints_left_pair, imgpoints_left_pair, gray.shape[::-1], None, None)


# In[27]:


# Stereo kalibracja
stereo_calibration_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)
stereo_calibration_flags = cv.CALIB_FIX_INTRINSIC

stereo_calibration_ret_val, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv.stereoCalibrate(objpoints_left_pair, imgpoints_left_pair, imgpoints_right_pair, mtx_left_pair, dist_left_pair, mtx_right_pair, dist_right_pair, gray.shape[::-1], criteria=stereo_calibration_criteria, flags=stereo_calibration_flags)

print("Stereo calibration")
print("Ret val:", stereo_calibration_ret_val)
print("cameraMatrix1:", cameraMatrix1)
print("distCoeffs1:", distCoeffs1)
print("cameraMatrix2:", cameraMatrix2)
print("distCoeffss2:", distCoeffs2)
print("R:", R)
print("T:", T)
print("E:", E)
print("F:", F)


# In[28]:


json_dump_pair = json.dumps({'stereo_calibration_ret_val': stereo_calibration_ret_val, 'cameraMatrix1': cameraMatrix1 , 'distCoeffs1': distCoeffs1 , 'cameraMatrix2': cameraMatrix2, 'distCoeffss2': distCoeffs2, 'R': R , 'T': T, 'E': E, 'F': F}, cls=NumpyEncoder)


with open('dataPair_k0.json', 'w') as outfile:
    json.dump(json_dump_pair, outfile)


# In[29]:


#Stereo rectification process


# In[30]:


rectify_scale = 1 # 0=full crop, 1=no crop
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, gray.shape[::-1], R, T, alpha = rectify_scale)
left_maps = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray.shape[::-1], cv.CV_16SC2)
right_maps = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray.shape[::-1], cv.CV_16SC2)


# In[31]:


left_img = cv.imread('k0/left_63.png')
right_img = cv.imread('k0/right_63.png')

left_img_remap = cv.remap(left_img, left_maps[0], left_maps[1], cv.INTER_LANCZOS4)
right_img_remap = cv.remap(right_img, right_maps[0], right_maps[1], cv.INTER_LANCZOS4)


# In[32]:


baseline = np.linalg.norm(T)

print(baseline) #mm 94.03398730787869


# In[33]:


cv.imwrite('rectification_right_63_k0.png', left_img_remap)
cv.imwrite('rectification_left_63_k0.png', right_img_remap)


# In[ ]:




