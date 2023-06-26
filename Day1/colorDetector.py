import cv2
import numpy as np

cap=cv2.VideoCapture(0)
lwr_blue=np.array([100,100,100])
upper_blue=np.array([10,255,255])
lwr_red=np.array([0,100,100])

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(frame,lwr_blue,upper_blue)
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.dilate(mask,kernel,iterations=1)
    res=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    cv2.imshow("Output",frame)
    cv2.imshow("Mask",mask)

    key=cv2.waitKey(30)
    if key==32:
        break

cap.release()
cv2.destroyAllWindows()