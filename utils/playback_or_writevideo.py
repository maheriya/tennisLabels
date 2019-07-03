#!/usr/bin/env python3
#
# Script can be used to simply playback 'video' of a *.jpg dump of a video
# The secondary purpose is to write a video as an .avi file

import os
import sys
import cv2 as cv
from glob import glob

## Do you want to monitor what is being written?
PLAYBACK = True

## Video dump directory:
IMGPATH = 'VideoStreams/frames-left'
if not os.path.isdir(IMGPATH):
    print("Video dump dir {} doesn't exist".format(IMGPATH))
    sys.exit(1)


VIDFILE = os.path.join(".", "tennis.avi")
#fourcc = cv.VideoWriter_fourcc(*'XVID')
fourcc = cv.VideoWriter_fourcc(*'DIVX')  ## DVIX and XVID are tested. Use vlc player to view... KM
writer = cv.VideoWriter(VIDFILE, fourcc, 60, (1920, 1080))
if not writer.isOpened():
    print("Could not create video writer")
    sys.exit(1)
else:
    print("Created video writer to write {} video file".format(VIDFILE))

## Browse through a source directory of video dump and create an AVI file
imgs = glob('{}/*.jpg'.format(IMGPATH))
imgs.sort()
for imgfile in imgs:
    img = cv.imread(imgfile, 1)
    writer.write(img)
    if PLAYBACK:
        cv.imshow('image', img)
        key = cv.waitKey(5) & 255
        if key == 27:
            break

writer.release()

if PLAYBACK:
    cv.destroyAllWindows()
