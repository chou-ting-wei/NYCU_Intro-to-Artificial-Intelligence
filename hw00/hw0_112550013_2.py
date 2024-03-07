import cv2
import numpy as np

vpath = 'HW0/data/video.mp4'

vidcap = cv2.VideoCapture(vpath)
_, lst_img = vidcap.read()
flg, now_img = vidcap.read()
cnt = 0

while flg:
    res_img = cv2.absdiff(now_img, lst_img)
    res_img[:, :, 0] = 0
    res_img[:, :, 2] = 0
    
    if cnt == 20:
        cv2.imwrite('HW0/hw0_112550013_2.png', np.hstack((now_img, res_img)))
    
    lst_img = now_img
    flg, now_img = vidcap.read()
    cnt += 1
    