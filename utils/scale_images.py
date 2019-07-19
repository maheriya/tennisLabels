#!/usr/bin/env python
# coding: utf-8

## Scale Images in a Directory
# This script scales all images in a VOCdevkit dataset to a specified output dimensions and
# updates the existing labels accordingly. The Pascal VOC annotations use pixel values for
# bounding boxes, and as a result, need to be scaled with the same scale as the images.
#
## Why Scale?
# We annotate the images in full resolution since it is easier to find shapes and bounding
# boxes can be more accurately defined. However, for training we don't need the images at full
# resolution. Also, carrying around images at full resolution requires a lot of disk space.
# Especially if we have to use cloud resources.
# 
# That is why we scale the images to the dimensions we need for training. Keeping it slightly
# larger then actual size needed for training is optimal because that allows using the random
# crop augmentation more effectively. Also, it saves time during training since images will
# not have to be scaled during training.

import os
import sys
import cv2 as cv
from lxml import etree
from lxml import objectify
from glob import glob
import subprocess


if sys.version_info[0] < 3:
    PYVER = 2
else:
    PYVER = 3


## Global variables
SHOW_IMAGES = False

def getImageSizefromAnnotations(annfile):
    ## Load annotations file
    with open(annfile) as f:
        xml = f.read()
    ann = objectify.fromstring(xml)

    ## Get the image size from annotation
    return (ann.size.width, ann.size.height)

def readAndScaleAnnotations(i_anndir, imgbase, SCALE):
    ## Load annotations file
    i_annfile = os.path.join(i_anndir, imgbase + ".xml")
    #print("Input annotation file: {}".format(i_annfile))
    with open(i_annfile) as f:
        xml = f.read()
    ann = objectify.fromstring(xml)

    ## Change the size based on scale
    ann.size.width  = objectify.StringElement(str(int(ann.size.width  * SCALE)))
    ann.size.height = objectify.StringElement(str(int(ann.size.height * SCALE)))
    folder = ann.folder
    filename = ann.filename
    filepath = os.path.join(str(folder), 'JPEGImages', str(filename))
    ann.path = objectify.StringElement(filepath)
    for obj in ann.iter('object'):
        obj.bndbox.ymin = objectify.StringElement(str(obj.bndbox.ymin * SCALE))
        obj.bndbox.xmin = objectify.StringElement(str(obj.bndbox.xmin * SCALE))
        obj.bndbox.ymax = objectify.StringElement(str(obj.bndbox.ymax * SCALE))
        obj.bndbox.xmax = objectify.StringElement(str(obj.bndbox.xmax * SCALE))

    return ann

def show_imgs(cvimg, cvimg_scaled):
    global SHOW_IMAGES
    cv.imshow("Original image", cvimg)
    cv.imshow("Scaled image", cvimg_scaled)
    key = cv.waitKey(0) & 255
    if key == 27:
        cv.destroyAllWindows()
        sys.exit(0)
    elif key == ord('g'): ## Go for it; don't show images after this
        cv.destroyAllWindows()
        SHOW_IMAGES = False


##-#####################################################################################
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "invoc", type=str, #default="/IMAGESETS/TENNIS/VOCdevkit",
    help="The input VOC root directory."
)
parser.add_argument(
    "outvoc", type=str, #default="/IMAGESETS/TENNIS/VOCdevkitScaled",
    help="Output VOC root directory."
)
parser.add_argument(
    "--height", type=float, default=540, required=False,
    help="Output image height. "
)
args = parser.parse_args()

# Select a size that is suitable for training.
# 
# Assume our input images are 1920x1080. In that case,
# Suggested sizes are:
# 480x270 - Scale of 1/4   
# 533x300 - Scale of 1/3.6 
# 640x360 - Scale of 1/3   
# 960x540 - Scale of 1/2   
# 
# Short edge of the input video size (first frame) will be used for determining the scale.
#
# Tennis ball detection is a difficult object detection problem. At normal resolutions (x270, x300 or x360),
# the ball is only a few pixel in diameter. At service line, it is 3 pixels in diameter!
# As a result, detection will be tough.  In the end, we have to run training experiments to find the optimal
# image size -- it is a speed vs accuracy tradeoff.

## Main variables
IN_VOCDIR = os.path.abspath(args.invoc)
IN_IMGDIR = os.path.join(IN_VOCDIR, "{}", "JPEGImages")   # Template
IN_ANNDIR = os.path.join(IN_VOCDIR, "{}", "Annotations")  # Template

OUT_VOCDIR = os.path.abspath(args.outvoc)
OUT_IMGDIR = os.path.join(OUT_VOCDIR, "{}", "JPEGImages") # Template
OUT_ANNDIR = os.path.join(OUT_VOCDIR, "{}", "Annotations")# Template

## Each directory under *_VOCDIR is a base dataset
findtask = subprocess.Popen(
    [r"find {}/ -mindepth 3 -name '*.xml' | sed -e 's#/Annotations/.*.xml##g' | sort | uniq".format(IN_VOCDIR)], 
     shell=True, stdout=subprocess.PIPE)
output,err = findtask.communicate()
if PYVER>=3:
    output = bytes.decode(output)

output = output.rstrip().split('\n')
vocbases = [os.path.basename(d) for d in output]
print(vocbases)
print(len(vocbases))


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

    ## Check by loading one image. For debug
    anns = glob("{}/*.xml".format(i_anndir))

    ## Determine scale -- for each VOC db. Some may be 1920x1080, some may be 4k
    (width, height) = getImageSizefromAnnotations(anns[0])
    SCALE = float(args.height) / float(height)
    print("Scale: {:.3f}".format(SCALE))


    for annfile in anns:
        imgbase = os.path.splitext(os.path.basename(annfile))[0]
        #print("  Image base {}".format(imgbase))

        ## Scale annotations
        ann = readAndScaleAnnotations(i_anndir, imgbase, SCALE)
        img_width = int(ann.size.width)
        img_height = int(ann.size.height)
        img_depth  = int(ann.size.depth)

        ## Load image
        cvimg = cv.imread(os.path.join(i_imgdir, imgbase+".jpg"), 1)
        ## Scale image
        cvimg_n = cv.resize(cvimg, (img_width, img_height), interpolation = cv.INTER_CUBIC)


        if SHOW_IMAGES:
            print("xmin: {}".format(ann.findall(".//xmin")))
            print("ymin: {}".format(ann.findall(".//ymin")))
            print("xmax: {}".format(ann.findall(".//xmax")))
            print("ymax: {}".format(ann.findall(".//ymax")))
            print("Scaled properties: width: {:4d}, height: {:4d}, depth: {:1d}".format(img_width, img_height, img_depth))
            # Check images
            show_imgs(cvimg, cvimg_n)

        obj_xml = etree.tostring(ann, pretty_print=True, xml_declaration=False)
        o_annfile = os.path.join(o_anndir, imgbase + ".xml")
        with open(o_annfile, 'w') as f:
            if PYVER>=3:
                f.write(obj_xml.decode('utf8'))
            else:
                f.write(obj_xml)

        o_imgfile = os.path.join(o_imgdir, imgbase+".jpg")
        #print("Writing scaled image {}".format(o_imgfile))
        cv.imwrite(o_imgfile, cvimg_n)

        cnt += 1

print("Done. Scaled {} annotations and images".format(cnt))

