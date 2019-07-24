#!/usr/bin/env python
#
# Visualize annotations


from __future__ import print_function
import os
import sys
import cv2 as cv

from glob import glob
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tennis_common as tc

def show_imgs(cvimg, zone=None):
    if zone is not None:
        imgstr = "Image Zone {}".format(zone)
    cv.imshow(imgstr, cvimg)
        
    key = cv.waitKey(0) & 255
    if key == 27:
        cv.destroyAllWindows()
        sys.exit(0)


def parseArgs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "invoc", type=str, #default="/IMAGESETS/TENNIS/VOCdevkit",
        help="The input VOC base or year directory (that contains JPEGImages folder)."
    )

    args = parser.parse_args()
    return args




##-#####################################################################################
args = parseArgs()
## Main variables
vocbase = os.path.abspath(args.invoc)

if 1:
    print("VOC Base: {}".format(vocbase))
    i_imgdir = os.path.join(vocbase, "JPEGImages")
    i_anndir = os.path.join(vocbase, "Annotations")
    if not os.path.isdir(i_imgdir):
        print("Input image dir {} is not accessible".format(i_imgdir))
    if not os.path.isdir(i_anndir):
        print("Input annotations dir {} is not accessible".format(i_anndir))

    annnames = glob("{}/*.xml".format(i_anndir))
    annnames = [os.path.basename(i) for i in annnames]
    annnames.sort() # Sort files to pick frames in order. It is assumed that xml/images are named likewise
    for annfile in annnames:
        try: zone = int(re.sub(r'.*-z(\d+).xml', r'\1', annfile))
        except: zone = None
        
        ann = tc.getAnnotations(os.path.join(i_anndir, annfile))
        imgfile = re.sub(r'.xml', '.jpg', annfile)
        cvimg = cv.imread(os.path.join(i_imgdir, os.path.join(i_imgdir, imgfile)), cv.IMREAD_COLOR)

        for obj in ann.iter('object'):
            bbox = [obj.bndbox.ymin, obj.bndbox.xmin, obj.bndbox.ymax, obj.bndbox.xmax]
            cvimg = tc.drawBoundingBox(cvimg, bbox, tc.LBL_IDS[obj.name])

        show_imgs(cvimg, zone)

cv.destroyAllWindows()


