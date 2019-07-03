#!/usr/bin/env python
# coding: utf-8
#
## Why Motion Dataset?
# There are a lot of object detection networks out there which are fine tuned for single image
# detection. These networks don't take motion into account and cannot learn from motion. If you
# consider tennis videos, and if you want to detect the tennis ball -- which is quite small and
# moves fast -- it is difficult to do with simple object detectors. There are networks or
# frameworks that are built to input multiple images for training as well as inference; however,
# they are overly complex and unsuitable for real-time inference on embedded devices as they
# typically combine inference of multiple images in one way or the other.
# 
# Our motion dataset goes for the low hanging fruit: it can work with any existing single-image
# based object detector without requiring full retraining from scratch. The motion information
# is embeded into the three channels that usually are reserved for RGB color information.
# 
# Advantage of this meothod is that the network inference speed remains the same even with the
# added learning from motion. This is quite suitable for motion based learning and inference.
# The accuracy of the network will be lower for stationary images (compared to normal networks)
# since we dont' use full RGB color information in this dataset.
#
## Create a Motion Dataset Suitable for Real-time Inference
# We convert a VOC dataset that is based on videos into motion dataset.
# There are two versions of motion dataset: 3FRAMES and FRAMESDIFF.
#
### 3FRAMES:
# For this particular version, we take 3 frames of video at a time, convert them into grascale
# as channels and finally, combine these three frame-channels into a single image that contains
# motion information.
#
### FRAMESDIFF:
# In this version, we keep red and green channels intact while replacing the blue channel with
# motion info channel. Motion info channel is constructed by combining the frame differences of
# three frames to create a trail of three objects.
# motion channel = bitwise_or(abs(curr - prev1), abs(prev1-prev2)).
# The reason for keeping all three objects (instead of, e.g., eliminating the other two) is to
# provide the network with the history of movement. This way, when the current frame is missing
# the object -- or is hard to detect -- the previous frame info can help
# This is different compared to CV method where you want to only keep the object in current
# current frame. For reference, that method is as follows (we don't use that):
#  cv_algo_channel = bitwise_and(abs(curr-prev1), abs(curr-prev2)).
# 
# 

from __future__ import print_function
import os
import sys
import cv2 as cv

from lxml import etree
from lxml import objectify
from glob import glob
import subprocess
import re
import shutil

if sys.version_info[0] < 3:
    PYVER = 2
else:
    PYVER = 3


## Select motion type
## MOTION DB setting: '3FRAMES' or 'FRAMESDIFF'
#MOTION_TYPE = '3FRAMES'
MOTION_TYPE = 'FRAMESDIFF'

## Select Number of Frames to Combine
# The simplest is to combine or pack three frames into three channels of the image. This will not require any changes to the network when re-training. If we choose more than three images, the input dimensions change, and that will require net-surgery of the network.

NFRAMES = 3
## Change this to view images
SHOW_IMAGES = False


def show_imgs(cvimg, cvimg_n):
    global SHOW_IMAGES
    cv.imshow("Original image", cvimg)
    cv.imshow("Motion image", cvimg_n)
    key = cv.waitKey(0) & 255
    cv.destroyAllWindows()
    if key == 27:
        #sys.exit(0)
        assert(False), "Exit requested"
    elif key == ord('g'): ## Go for it; don't show images after this
        SHOW_IMAGES = False

def getNumberingScheme(imgname):
    fnum     = re.sub(r'.*[-_](\d+).jpg', r'\1', imgname)
    fpre     = re.sub(r'(.*[-_])(\d+).jpg', r'\1', imgname)
    numlen   = len(fnum)
    numtmplt = '{:0' + str(numlen) + 'd}'
    return (fpre, numtmplt)




##-#####################################################################################
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "invoc", type=str, #default="/IMAGESETS/TENNIS/VOCdevkitScaled",
    help="The input VOC root directory."
)
parser.add_argument(
    "outvoc", type=str, #default="/IMAGESETS/TENNIS/VOCdevkitMotion",
    help="Output VOC root directory."
)
args = parser.parse_args()


## Main variables
IN_VOCDIR = os.path.abspath(args.invoc)
IN_IMGDIR = os.path.join(IN_VOCDIR, "{}", "JPEGImages")   # Template
IN_ANNDIR = os.path.join(IN_VOCDIR, "{}", "Annotations")  # Template

OUT_VOCDIR = os.path.abspath(args.outvoc)
OUT_IMGDIR = os.path.join(OUT_VOCDIR, "{}", "JPEGImages") # Template
OUT_ANNDIR = os.path.join(OUT_VOCDIR, "{}", "Annotations")# Template

## Find base datasets containing annotations
findtask = subprocess.Popen(
    [r"find {}/ -mindepth 3 -name '*.xml' | sed -e 's#/Annotations/.*.xml##g' | sort | uniq".format(IN_VOCDIR)], 
     shell=True, stdout=subprocess.PIPE)
output,err = findtask.communicate()
if PYVER<3:
    output = output.rstrip().split('\n')
else:
    output = (bytes.decode(output).rstrip()).split('\n')
vocbases = [os.path.basename(d) for d in output]
print(vocbases)
print("There are {} datasets to process".format(len(vocbases)))


cnt = 0
for base in vocbases:
    print("VOC Base: {}".format(base))
    i_imgdir = IN_IMGDIR.format(base)
    i_anndir = IN_ANNDIR.format(base)
    if not os.path.isdir(i_imgdir):
        print("Input image dir {} is not accessible".format(i_imgdir))
    if not os.path.isdir(i_anndir):
        print("Input annotations dir {} is not accessible".format(i_anndir))

    o_imgdir = OUT_IMGDIR.format(base)
    o_anndir = OUT_ANNDIR.format(base)
    for dir in [o_imgdir, o_anndir]:
        if not os.path.isdir(dir):
            os.makedirs(dir)
        else:
            print("Dir {} already exists".format(dir))

    ## Create image list to process
    imgs = glob("{}/*.jpg".format(i_imgdir))
    imgs = [os.path.basename(i) for i in imgs]
    imgs.sort() # Sort images to pick frames in order. It is assumed the images are named likewise

    (fprefix, ntemplate) = getNumberingScheme(imgs[0])
    #print("fprefix: {}, template: {}".format(fprefix, ntemplate))

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(4,4))
    i = 2  ## Index
    for img in imgs[2:]:
        imgbase = os.path.splitext(os.path.basename(img))[0]

        img_i  = imgs[i]
        img_i1 = imgs[i-1]
        img_i2 = imgs[i-2]
        i += 1

        fnum    = int(re.sub(r'.*[-_](\d+).jpg', r'\1', img_i))
        eimg_i  = fprefix + ntemplate.format(fnum) + '.jpg'
        eimg_i1 = fprefix + ntemplate.format(fnum-1) + '.jpg'
        eimg_i2 = fprefix + ntemplate.format(fnum-2) + '.jpg'
        if img_i != eimg_i or img_i1 != eimg_i1 or img_i2 != eimg_i2:
            # Not a continuous series of three frames including previous two, we skip this frame
            print("Skipping {}".format(img_i))
            continue


        ## load images as grayscale
        cvimg_c= cv.imread(os.path.join(i_imgdir, img_i), cv.IMREAD_COLOR)
        cvimg  = cv.cvtColor(cvimg_c, cv.COLOR_BGR2GRAY)
        cvimg1 = cv.imread(os.path.join(i_imgdir, img_i1), cv.IMREAD_GRAYSCALE)
        cvimg2 = cv.imread(os.path.join(i_imgdir, img_i2), cv.IMREAD_GRAYSCALE)
        ## Create frame-diff based background subtracted image with a trail of three balls
        ## We are doing this (keeping the trail) on purpse. This to provide the network
        ## with some referene in the case when the ball is not visible in the current frame
        ## but it was visible in previous frames.
        diff_p1p2 = cv.absdiff(cvimg1, cvimg2)
        diff_cp1  = cv.absdiff(cvimg, cvimg1)
        image_b   = cv.bitwise_or(diff_p1p2, diff_cp1) ## This will create the trail of three objects
        image_diff= cv.dilate(image_b, kernel) ## enlarge the blobs

        if MOTION_TYPE == '3FRAMES':
            # Merge (merge 3 grascale motion frames into BGR channels)
            cvimg_n  = cv.merge([cvimg, cvimg1, cvimg2])
        elif MOTION_TYPE == 'FRAMESDIFF':
            # Replace blue channel with frame diff. Blue channel is less important in tennis for us
            # since the ball is greenish yellow -- most information in red and green channel.
            cvimg_c[:,:,0] = image_diff
            cvimg_n = cvimg_c
        else:
            print("Unsupported motion type {}".format(MOTION_TYPE))
            sys.exit(1)

        if SHOW_IMAGES:
            # Check images
            show_imgs(cvimg, cvimg_n)

        ## Copy annoation file
        i_annfile = os.path.join(i_anndir, imgbase + ".xml")
        o_annfile = os.path.join(o_anndir, imgbase + ".xml")
        shutil.copy(i_annfile, o_annfile)
        
        ## Write new combined motion image
        o_imgfile = os.path.join(o_imgdir, imgbase+".jpg")
        cv.imwrite(o_imgfile, cvimg_n)
        #if (cnt >= 10):
        #    assert(False), "Temp exit"
            
        cnt += 1

print("Done. Motion Dataset created with {} annotations and images".format(cnt))

