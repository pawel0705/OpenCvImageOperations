# Rzadki przepływ optyczny - metoda Lucas-Kanade.

import numpy as np
import cv2 as cv

cap = cv.VideoCapture('s3/left_%01d.png')

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 10,
                       blockSize = 10)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))
ret, old_frame = cap.read()
p0 = cv.goodFeaturesToTrack(old_frame, mask = None, **feature_params)
mask = np.zeros_like(old_frame)

while(1):
    ret, frame = cap.read()
    if not ret:
        break
        
    p1, st, err = cv.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    cv.imshow('img', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    old_gray = frame.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()


# In[ ]:




