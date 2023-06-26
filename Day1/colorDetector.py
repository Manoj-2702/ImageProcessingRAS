import cv2
import numpy as np

cap=cv2.VideoCapture(0)
lwr_blue=np.array([100,100,100])
upper_blue=np.array([10,255,255])

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(30)
    if key==32:
        break

cap.release()
cv2.destroyAllWindows()