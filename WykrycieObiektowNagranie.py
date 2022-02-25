# Metoda Gurner-Farneback, gęsty przepływ optyczny. Wykrywanie obiektów na nagraniu.


import math
import numpy as np
import cv2 as cv

class RectangleTracker:
    def __init__(self):
        self.rectanle_center = {}
        self.id_rectangle = 0


    def getRectangle(self, objects_rect):
        objects_bounding_box = []

        for rect in objects_rect:
            x, y, w, h = rect
            cy = (h + y + y) // 2
            cx = (w + x + x) // 2

            same_object = False
            for id, point in self.rectanle_center.items():
                dist = math.hypot(cx - point[0], cy - point[1])

                if dist < 30:
                    self.rectanle_center[id] = (cx, cy)
                    objects_bounding_box.append([x, y, w, h, cx, cy, point[0], point[1], id])
                    same_object = True
                    break

            if same_object == False:
                self.rectanle_center[self.id_rectangle] = (cx, cy)
                objects_bounding_box.append([x, y, w, h, cx, cy, cx, cy, self.id_rectangle])
                self.id_rectangle += 1

        rectanle_center = {}
        for obj_bb_id in objects_bounding_box:
            _, _, _, _, _, _, _, _, id_obj = obj_bb_id
            center_tmp = self.rectanle_center[id_obj]
            rectanle_center[id_obj] = center_tmp

        self.rectanle_center = rectanle_center.copy()
        return objects_bounding_box
    
    
trackerRect = RectangleTracker()

cap = cv.VideoCapture("vtest.avi")

ret, first_frame = cap.read()
  
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
  
mask = np.zeros_like(first_frame)

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, 
                                       None,
                                       0.5, 3, 15, 6, 3, 1.2, 1)
      
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    _, mask = cv.threshold(mag, 1, 255, cv.ADAPTIVE_THRESH_MEAN_C)
    mask8bit = cv.normalize(mask, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    contours, _ = cv.findContours(mask8bit, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 200 and area < 10000:
            x, y, w, h = cv.boundingRect(cnt)
            detections.append([x, y, w, h])

    boxes_ids = trackerRect.getRectangle(detections)
    for box_id in boxes_ids:
        x, y, w, h, newCx, newCy, oldCx, oldCy, id = box_id
        distance = ((((oldCx - newCx)**2) + ((oldCy-newCy)**2) )**0.5)
        color = (0, 255, 0)
        if distance > 8:
            color = (0, 0, 255)
            
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv.arrowedLine(frame, (oldCx,oldCy), (newCx,newCy), color, 2, tipLength=0.5)
    
    cv.imshow("input", frame)
      
    prev_gray = gray
      
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
        
cap.release()
cv.destroyAllWindows()


# In[ ]:




