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
# Latest:
# A. Data mining: Find consecutive frames containing ball labels (don't care about racket).
#    Until now we only looked for sequence of images without looking at actual labels. Now,
#    we consider only those frame sequences for training that all have ball labels.
# B. Use two trailing ball bboxes, as a second label. Take a union of these two bboxes. The
#    intention is to task the neural network with extra detection duties during training. The 
#    hope is that by doing this, the network will be forced to learn extra motion information.
#    and at inference time, we cab utilize this bbox to determine the direction of the ball.
# The above A and B are independent additions; either can be implemented without the other.


from __future__ import print_function
import os
import sys
import cv2 as cv

from glob import glob
import subprocess
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tennis_common as tc


## MOTION DB setting: '3FRAMES' or 'FRAMESDIFF'
MOTION_TYPE = 'FRAMESDIFF'

## Change this to view images
SHOW_IMAGES = False

## Verbosity
DEBUG = 0
tc.DEBUG = DEBUG

def show_imgs(cvimg, cvimg_n):
    global SHOW_IMAGES
    cv.imshow("Original image", cvimg)
    cv.imshow("Motion image", cvimg_n)
    key = cv.waitKey(0) & 255
    if key == 27:
        cv.destroyAllWindows()
        sys.exit(0)
    elif key == ord('g'): ## Go for it; don't show images after this
        cv.destroyAllWindows()
        SHOW_IMAGES = False


##-#####################################################################################
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
output = tc.runSystemCmd(r"find {}/ -mindepth 3 -name '*.xml' | sed -e 's#/Annotations/.*.xml##g' | sort | uniq".format(IN_VOCDIR))
vocbases = [os.path.basename(d) for d in output]
#print(vocbases)
print("There are {} datasets to process".format(len(vocbases)))


cnt = 0
for vocbase in vocbases:
    print("VOC Base: {}".format(vocbase))
    i_imgdir = IN_IMGDIR.format(vocbase)
    i_anndir = IN_ANNDIR.format(vocbase)
    if not os.path.isdir(i_imgdir):
        print("Input image dir {} is not accessible".format(i_imgdir))
    if not os.path.isdir(i_anndir):
        print("Input annotations dir {} is not accessible".format(i_anndir))

    o_imgdir = OUT_IMGDIR.format(vocbase)
    o_anndir = OUT_ANNDIR.format(vocbase)
    for idir in [o_imgdir, o_anndir]:
        if not os.path.isdir(idir):
            os.makedirs(idir)
        else:
            print("Dir {} already exists".format(idir))

    ## Create image list to process
    imgs = glob("{}/*.jpg".format(i_imgdir))
    imgs = [os.path.basename(i) for i in imgs]
    imgs.sort() # Sort images to pick frames in order. It is assumed the images are named likewise
    (fprefix, ntemplate) = tc.getNumberingScheme(imgs[0])


    annnames = glob("{}/*.xml".format(i_anndir))
    annnames = [os.path.basename(i) for i in annnames]
    annnames.sort() # Sort files to pick frames in order. It is assumed that xml/images are named likewise
    if len(annnames) < 3:
        print("This VOC Base has less than 3 annotations. Skipping.")
        continue

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(4,4))
    i = 2  ## Index
    for annfile in annnames[2:]:
        img_i  = imgs[i]
        img_p1 = imgs[i-1]
        img_p2 = imgs[i-2]
        i += 1

        fnum    = int(re.sub(r'.*[-_](\d+).jpg', r'\1', img_i))
        eimg_i  = fprefix + ntemplate.format(fnum) + '.jpg'
        eimg_p1 = fprefix + ntemplate.format(fnum-1) + '.jpg'
        eimg_p2 = fprefix + ntemplate.format(fnum-2) + '.jpg'
        if img_i != eimg_i or img_p1 != eimg_p1 or img_p2 != eimg_p2:
            # Not a continuous series of three frames including previous two, we skip this frame
            if DEBUG>=1:
                print("Skipping. Frame sequence not found for {}. ".format(img_i))
            continue  # Get next image/ann
        else:
            if DEBUG>=1:
                print("Processing {}".format(img_i))

        ## Now that we found three sequential frames, let's check if they all have ball labels
        annfiles = [fprefix + ntemplate.format(fn) + '.xml' for fn in [fnum, fnum-1, fnum-2]]
        anns = [tc.getAnnotations(os.path.join(i_anndir, annfile)) for annfile in annfiles]
        seq = True
        for ann_ in anns:
            objs = ann_.findall('.//object/name')
            if 'ball' not in objs:
                seq = False
                break # don't check other anns
        if not seq:
            if DEBUG>=1:
                print("\tSkipping. 3 ball labels sequence not found for {}".format(img_i))
            continue # Get next image/ann

        ballUBox, _ = tc.getUBoxes(anns[1:]) # Find union bbox for ball label from two previous frames
        assert(ballUBox is not None),"Error! Cannot find union of previous two balls bounding boxes"
        ## Add this as a new label. We call this label 'pballs' for 'previous balls'
        tc.addAnnotation(anns[0], 'pballs', ballUBox)

        ## load images
        cvimg_c= cv.imread(os.path.join(i_imgdir, img_i), cv.IMREAD_COLOR)
        cvimg  = cv.cvtColor(cvimg_c, cv.COLOR_BGR2GRAY)
        cvimg1 = cv.imread(os.path.join(i_imgdir, img_p1), cv.IMREAD_GRAYSCALE)
        cvimg2 = cv.imread(os.path.join(i_imgdir, img_p2), cv.IMREAD_GRAYSCALE)

        if MOTION_TYPE == '3FRAMES':
            # Merge 3 grayscale motion frames into BGR channels)
            cvimg_n  = cv.merge([cvimg, cvimg1, cvimg2])
        elif MOTION_TYPE == 'FRAMESDIFF':
            ## Create frame-diff based background subtracted image with a trail of three balls
            ## We are doing this (keeping the trail) on purpse. This to provide the network
            ## with some referene in the case when the ball is not visible in the current frame
            ## but it was visible in previous frames.
            diff_p1p2 = cv.absdiff(cvimg1, cvimg2)
            diff_cp1  = cv.absdiff(cvimg, cvimg1)
            image_b   = cv.bitwise_or(diff_p1p2, diff_cp1) ## This will create the trail of three objects
            #bring back? =>#image_diff= cv.dilate(image_b, kernel) ## enlarge the blobs
            # Replace blue channel with frame diff. Blue channel is less important in tennis for us
            # since the ball is greenish yellow -- most information in red and green channel.
            cvimg_n = cvimg_c.copy()
            cvimg_n[:,:,0] = image_b #image_diff
        else:
            print("Unsupported motion type {}".format(MOTION_TYPE))
            sys.exit(1)

        ######################################################################################
        ## Write output files
        ######################################################################################

        ## Write annotation file 
        tc.cleanUpAnnotations(anns[0], ['ball', 'racket', 'pballs'])
        tc.writeAnnotation(anns[0], os.path.join(o_anndir, annfiles[0]))
        
        ## Write new motion image
        cv.imwrite(os.path.join(o_imgdir, img_i), cvimg_n)
            
        if SHOW_IMAGES:
            for obj in anns[0].iter('object'):
                bbox = [obj.bndbox.ymin, obj.bndbox.xmin, obj.bndbox.ymax, obj.bndbox.xmax]
                cvimg_n = tc.drawBoundingBox(cvimg_n, bbox, tc.LBL_IDS[obj.name])

            # show annotated images
            show_imgs(cvimg, cvimg_n)

        cnt += 1

cv.destroyAllWindows()
print("Done. Motion Dataset created with {} annotations and images".format(cnt))

