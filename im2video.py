import os
import numpy as np
import cv2

imgl=os.listdir("./imgs")
fourcc = cv2.VideoWriter_fourcc(*'MPV4')
videoWriter = cv2.VideoWriter('saveVideo.mp4', fourcc, 30, (1024,512))
for l in imgl:
    im=cv2.imread(os.path.join("./imgs",l))
    videoWriter.write(im)
videoWriter.release()

