#!/usr/bin/env python3
#
# Script can be used to simply playback 'video' of a *.jpg dump of a video
# The secondary purpose is to write a video as an .avi file

import os
import sys
import cv2 as cv
from glob import glob
import argparse

## Do you want to monitor what is being written?
PLAYBACK = True

def parseArgs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "inDir", type=str, 
        help="The input frames dump directory."
    )
    parser.add_argument(
        "aviFile", type=str,
        help="Output AVI file name path"
    )
    
    args = parser.parse_args()
    return args




##-#####################################################################################
args = parseArgs()
IMGPATH = args.inDir
VIDFILE = os.path.abspath(args.aviFile)
 
if not os.path.isdir(IMGPATH):
    print("Video dump dir {} doesn't exist".format(IMGPATH))
    sys.exit(1)


outdir = os.path.dirname(VIDFILE)
if not os.path.isdir(outdir):
    os.makedirs(outdir)

imgs = glob('{}/*.jpg'.format(IMGPATH))
imgs.sort()
img = cv.imread(imgs[0], 1)
height         = img.shape[0]
width          = img.shape[1]

#fourcc = cv.VideoWriter_fourcc(*'XVID')
fourcc = cv.VideoWriter_fourcc(*'DIVX')  ## DVIX and XVID are tested. Use vlc player to view... KM
writer = cv.VideoWriter(VIDFILE, fourcc, 60, (width, height))
if not writer.isOpened():
    print("Could not create video writer")
    sys.exit(1)
else:
    print("Created video writer to write {} video file".format(VIDFILE))

## Browse through a source directory of video dump and create an AVI file
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
