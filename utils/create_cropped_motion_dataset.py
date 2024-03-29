#!/usr/bin/env python
#
# Given a VOC dataset of TENNIS videos dumped at 1920x1080 resolution, this script creates a
# scaled and cropped dataset. Even though the cropped zone size is static (1280x720/640x360) 
# crop scale), the zones themselves are dynamically selected based on the objects locations
# (by reading the annotations). 
# The zone size 1280x720 is selected for multiple reasons: [Other size is 640x360]
#   a. This size (2/3 of full scale) gives grid boxes of 1/3rd the full scale. This grid size
#      is the minimum overlap between the diagonal zones. Horizontal and vertically aligned
#      zones have the overlap that is double the height or width of this grid size. The 
#      minimum grid size is large enough to include a trail of tennis ball across three frames
#      even at fast speeds. This allows us to fully utilize motion information during training.
#   b. When images are cropped at 1280x720, and then finally scaled by 1/2, we get 640x360
#      as the final image size. This works perfectly with either 533x300 or 300x300 of final
#      training size while still allowing for random crop for training time data augmentation.
#
# Alternative to 1280x720 cropping is direct cropping at 640x360. Of course, this imposes 
# stricter tracking requirement at inference time. 
#
# Since we want this to work well for motion dataset for at least three frames of motion, the
# algorithm reads three frames at a time to decide how to crop the images. The three frames of
# motion also adds inherent hysteresis to the zone selection, making it stable.
#
# The algorithm is as follows:
# 1. Read three sequential frames -- current, prev1, prev2
# 2. Read annotations. Use 'ball' and 'racket' objects annotations for zones selection.
# 3. Create a union of bboxes for each object across three frames. Let's call this uboxes.
# 4. Select zones to crop: The zone selection is based on how centered a ubox is inside a zone.
#    Since zones have significant overlap with each other, multiple zones may contain an 
#    object. We compute the distance of each ubox center from the center of the zone.
#    For each object, the zone where this distance is the smallest is selected.
# 5. Crop out the selected zone/s to create output image/s.
# 
# Note that here the emphasis is NOT to center the objects within the cropped output. If we did
# that, the network will incorrectly learn to expect the objects at the center of the image.
# Since we can't provide the network with such images at the inference time, this type of
# training will be useless.
# Instead, we use fixed, four zone locations within the image, and select the zones purely on
# the basis of how *close* an object is to a zone center. This method guarantees to create
# output images where the objects will be found in various locations within the image which
# adds a good amount of regularization to the training and avoid overfitting.
#
# For the real-time inference, the application must make an initial guess about which region
# to crop for the input to the network, and may require multiple tries in the beginning.
# However, once the ball is detected, the one can implement rudimentary tracking for the next
# crop. Since ball detection (and not the racket detection) is the most important part of
# detection, decision making is trivial.
#
# Just to be clear, it is not necessary to use the same zones during inference; any region
# within the image will be fine as long as it contains the ball. When the ball nears the
# player, the racket will automatically get into the view. Note that at the time of training,
# we utilize all available samples of racket images, not just the images where both ball and 
# racket are visible at the same time.


from __future__ import print_function
import os
import sys
import cv2 as cv

from lxml import etree
from glob import glob
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tennis_common as tc

## INPUT IMAGE DIMENSIONS (scaled to these dimensions if required)
WIDTH  = 1920
HEIGHT = 1080

## MOTION DB setting: '3FRAMES' or 'FRAMESDIFF'
MOTION_TYPE = 'FRAMESDIFF'

## Change this to view images
SHOW_IMAGES = False

## Verbosity
DEBUG = 0
tc.DEBUG = DEBUG

def show_imgs(cvimg, cvimg_n, oimgs=[]):
    global SHOW_IMAGES
    cv.imshow("Original image", cvimg)
    cv.imshow("Motion image", cvimg_n)
    s = ["Out 1", "Out 2"]
    for i in range(len(oimgs)):
        cv.imshow(s[i], oimgs[i])
        
    key = cv.waitKey(2) & 255
    if key == 27:
        cv.destroyAllWindows()
        sys.exit(0)
    elif key == ord('g'): ## Go for it; don't show images after this
        cv.destroyAllWindows()
        SHOW_IMAGES = False


def drawZone(img, zones, zid, cropsize):
    if (cropsize[1] == 720):
        ## This is a fixed -- hardcoded -- grid of 4 equal sized zones:
        # Zones: top-left, top-right, bottom-left, bottom-right
        h = img.shape[0]
        w = img.shape[1]
        gy = [0, int(h/3.), int(h*2.0/3.0), h]
        gx = [0, int(w/3.), int(w*2.0/3.0), w]
    
        if zid == 0:
            img = cv.rectangle(img, (gx[0], gy[0]), (gx[2], gy[2]-2), (255, 196, 128), 2) ## T-L
        elif zid == 1:
            img = cv.rectangle(img, (gx[1]+2, gy[0]), (gx[3], gy[2]), (128, 255, 128), 2) ## T-R
        elif zid == 2:
            img = cv.rectangle(img, (gx[0], gy[1]), (gx[2]+2, gy[3]), (255, 128, 0), 2)   ## B-L
        elif zid == 3:
            img = cv.rectangle(img, (gx[1], gy[1]+2), (gx[3], gy[3]), (196, 0, 255), 2)   ## B-R
        else:
            print("Zone {} is not supported".format(zid))
    else:
        colors = [(255, 196, 128), (128, 255, 128), (255, 128, 0), (196, 0, 255), (206, 206, 128), (128, 206, 255)]
        gy0,gx0,gy2,gx2 = [int(b) for b in zones.getBBox(zid)]
        img = cv.rectangle(img, (gx0, gy0), (gx2, gy2-2), colors[zid%6], 1)
    return img


def parseArgs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "invoc", type=str, #default="/IMAGESETS/TENNIS/VOCdevkit",
        help="The input VOC root directory."
    )
    parser.add_argument(
        "outvoc", type=str, #default="/IMAGESETS/TENNIS/VOCdevkitCropped",
        help="Output VOC root directory."
    )
    parser.add_argument(
        "--height", type=int, default=720, required=False,
        help="Output image height. Not used right now."
    )
    
    args = parser.parse_args()
    return args




##-#####################################################################################
args = parseArgs()
## Main variables
IN_VOCDIR = os.path.abspath(args.invoc)
IN_IMGDIR = os.path.join(IN_VOCDIR, "{}", "JPEGImages")   # Template
IN_ANNDIR = os.path.join(IN_VOCDIR, "{}", "Annotations")  # Template

OUT_VOCDIR = os.path.abspath(args.outvoc)
OUT_IMGDIR = os.path.join(OUT_VOCDIR, "{}", "JPEGImages") # Template
OUT_ANNDIR = os.path.join(OUT_VOCDIR, "{}", "Annotations")# Template

cropsize = (int(args.height*16./9.), args.height)
if args.height != 720 and args.height != 360:
    print("Crop height of {} is not supported (use 720 or 360).".format(args.height))
    sys.exit(1)

## Find base datasets containing annotations
output = tc.runSystemCmd(r"find {}/ -mindepth 3 -name '*.xml' | sed -e 's#/Annotations/.*.xml##g' | sort | uniq".format(IN_VOCDIR))
vocbases = [os.path.basename(d) for d in output]
#print(vocbases)
print("There are {} datasets to process".format(len(vocbases)))


cnt = 0
dbcnt = 0
for vocbase in vocbases:
    dbcnt += 1
    print("\n{}/{}. VOC Base: {}".format(dbcnt, len(vocbases), vocbase))
    print("-------------------------------------------------")
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

    if cropsize[1] == 720:
        ## Define the grid points
        ##   0/3                 1/3                 2/3         3/3
        gy = [0,        int(HEIGHT/3.), int(HEIGHT*2.0/3.0), HEIGHT]
        gx = [0,        int( WIDTH/3.), int( WIDTH*2.0/3.0),  WIDTH]
        ## Create zones based on the grid
        zones = tc.BoundingBoxes('zones')
        #              ymin   xmin   ymax   xmax
        zones.addBBox([gy[0], gx[0], gy[2], gx[2]])  # Top-left zone
        zones.addBBox([gy[0], gx[1], gy[2], gx[3]])  # Top-right zone
        zones.addBBox([gy[1], gx[0], gy[3], gx[2]])  # Bottom-left zone
        zones.addBBox([gy[1], gx[1], gy[3], gx[3]])  # Bottom-right zone
    else: # cropsize[1] == 360:
        ## Define the grid points
        ##   0/6          1/6                 2/6                 3/6                 4/6                  5/6          6/6
        gy = [0, int(HEIGHT/6.),     int(HEIGHT/3.),     int(HEIGHT/2.), int(HEIGHT*2.0/3.0), int(HEIGHT*5.0/6.0),  HEIGHT]
        gx = [0, int( WIDTH/6.),     int( WIDTH/3.),     int( WIDTH/2.), int( WIDTH*2.0/3.0), int( WIDTH*5.0/6.0),   WIDTH]
        ## Create zones based on the grid
        zones = tc.BoundingBoxes('zones')
        for y in range(len(gy)-2):
            for x in range(len(gx)-2):
                zones.addBBox([gy[y], gx[x], gy[y+2], gx[x+2]])

    annnames = glob("{}/*.xml".format(i_anndir))
    annnames = [os.path.basename(i) for i in annnames]
    annnames.sort() # Sort files to pick frames in order. It is assumed that xml/images are named likewise
    if len(annnames) < 3:
        print("This VOC Base has less than 3 annotations. Skipping.")
        continue

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(4,4))
    i = 2  ## Index
    for annfile in annnames[2:]:
        annName_i  = annnames[i]
        annName_p1 = annnames[i-1]
        annName_p2 = annnames[i-2]
        i += 1

        fnum    = int(re.sub(r'.*[-_](\d+).xml', r'\1', annName_i))
        eannName_i  = fprefix + ntemplate.format(fnum) + '.xml'
        eannName_p1 = fprefix + ntemplate.format(fnum-1) + '.xml'
        eannName_p2 = fprefix + ntemplate.format(fnum-2) + '.xml'
        if annName_i != eannName_i or annName_p1 != eannName_p1 or annName_p2 != eannName_p2:
            # Not a continuous series of three frames including previous two, we skip this frame
            if 1: #DEBUG>=1:
                print("Skipping. Frame sequence not found for {}. ".format(annName_i))
            continue  # Get next image/ann
        else:
            if DEBUG>=1:
                print("Processing {}".format(annName_i))

        ## Now that we got a three sequential frames, let's read annotations and get uboxes
        ## uboxes = union of bboxes for each of the 'ball' or 'racket' bbox in all three images
        ## We are assuming only one 'ball' annotation per image. However, it is easy to handle
        ## multiple balls per image too. Not needed for our dataset.
        annfiles = [fprefix + ntemplate.format(fn) + '.xml' for fn in [fnum, fnum-1, fnum-2]]
        anns = [tc.getAnnotations(os.path.join(i_anndir, annfile)) for annfile in annfiles]
        seq = True
        for ann_ in anns:
            objs = ann_.findall('.//object/name')
            if 'ball' not in objs:
                seq = False
                break # don't check other anns
        if not seq:
            if 1: # DEBUG>=1:
                print("\tSkipping. 3 ball labels sequence not found for {}".format(annName_i))
            continue # Get next image/ann
        ballUBox, _ = tc.getUBoxes(anns[1:]) # Find union bbox for ball label from two previous frames
        assert(ballUBox is not None),"Error! Cannot find union of previous two balls bounding boxes"
        ## Add this as a new label. We call this label 'pballs' for 'previous balls'
        tc.addAnnotation(anns[0], 'pballs', ballUBox)

        w = anns[0].size.width
        ## Scale input to WIDTHxHEIGHT fixed dimensions if input size is different
        if w != WIDTH:
            scale = float(WIDTH) / float(w)
            ## Scale annotations
            anns = [tc.scaleAnnotations(ann, scale) for ann in anns]
        else:
            scale = 1.0

        ballUBox, racketUBox = tc.getUBoxes(anns)
        ## Find best enclosing zone for ball and racket UBoxes
        zid_b = zones.findEnclosing(ballUBox)
        zid_r = zones.findEnclosing(racketUBox)
        crop_zids = []
        if zid_b == zid_r: ## Both ball and racket are in the same zone
            if zid_b is not None:
                crop_zids.append(zid_b)
        else:
            for zid in [zid_b, zid_r]:
                if zid is not None:
                    crop_zids.append(zid)
        if DEBUG>=1:
            print("Crop Zones: {}".format(crop_zids))
        #assert(len(crop_zids) != 0), "No zones found for cropping. This means that the frame doesn't have ball or racket"
        if len(crop_zids) == 0:
            print("No zones found for cropping. This means that the frame doesn't have ball or racket. Skipped")
            continue

        ## load images as grayscale
        img_i, img_p1, img_p2 = [fprefix + ntemplate.format(fn) + '.jpg' for fn in [fnum, fnum-1, fnum-2]]
        _cvimg_c = cv.imread(os.path.join(i_imgdir, img_i), cv.IMREAD_COLOR)
        _cvimg   = cv.cvtColor(_cvimg_c, cv.COLOR_BGR2GRAY)
        _cvimg1  = cv.imread(os.path.join(i_imgdir, img_p1), cv.IMREAD_GRAYSCALE)
        _cvimg2  = cv.imread(os.path.join(i_imgdir, img_p2), cv.IMREAD_GRAYSCALE)

        if w != WIDTH:
            ## Resize if scale is different
            cvimg_c = cv.resize(_cvimg_c, (WIDTH, HEIGHT), interpolation = cv.INTER_CUBIC)
            cvimg   = cv.resize(_cvimg,   (WIDTH, HEIGHT), interpolation = cv.INTER_CUBIC)
            cvimg1  = cv.resize(_cvimg1,  (WIDTH, HEIGHT), interpolation = cv.INTER_CUBIC)
            cvimg2  = cv.resize(_cvimg2,  (WIDTH, HEIGHT), interpolation = cv.INTER_CUBIC)
        else:
            cvimg_c = _cvimg_c
            cvimg   = _cvimg
            cvimg1  = _cvimg1
            cvimg2  = _cvimg2

        if MOTION_TYPE == '3FRAMES':
            # Merge (merge 3 grascale motion frames into BGR channels)
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

        ## Crop images and annotations as per selected zones
        imgfilenames = []
        annfilenames = []
        outimgs = []
        outanns = []
        for zid in crop_zids:
            imgbase = fprefix + ntemplate.format(fnum) + '-z{:02d}'.format(zid)
            imgname = imgbase + '.jpg'
            annname = imgbase + '.xml'
            imgfilenames.append(imgname)
            annfilenames.append(annname)
            roi = zones.getBBox(zid)
            outann = tc.cropAnnotations(anns[0], roi, imgname, 6)
            outimg = zones.getImgRoI(zid, cvimg_n).copy()
            outanns.append(outann)
            outimgs.append(outimg)
            if DEBUG>=3 and len(crop_zids) > 1:
                obj_xml = etree.tostring(outann, pretty_print=True, xml_declaration=False)
                print("Annotation {}\n{}".format(annname, obj_xml))


        ######################################################################################
        ## Write output files
        ######################################################################################

        for index in range(len(outimgs)):
            ## Write annotation files
            tc.cleanUpAnnotations(outanns[index], ['ball', 'racket', 'pballs'])
            tc.writeAnnotation(outanns[index], os.path.join(o_anndir, annfilenames[index]))

            ## Write cropped motion images
            imgfile = os.path.join(o_imgdir, imgfilenames[index])
            if DEBUG>=2:
                print("Writing {}".format(imgfile))
            cv.imwrite(imgfile, outimgs[index])

        if SHOW_IMAGES:
            for zid in crop_zids:
                cvimg_n = drawZone(cvimg_n, zones, zid, cropsize)
            for index in range(len(outimgs)):
                img = outimgs[index]
                for obj in outanns[index].iter('object'):
                    bbox = [obj.bndbox.ymin, obj.bndbox.xmin, obj.bndbox.ymax, obj.bndbox.xmax]
                    outimgs[index] = tc.drawBoundingBox(outimgs[index], bbox, tc.LBL_IDS[obj.name])

            ## Draw bounding boxes
            if ballUBox is not None:
                cvimg_n = tc.drawBoundingBox(cvimg_n, ballUBox, 1)
            if racketUBox is not None:
                cvimg_n = tc.drawBoundingBox(cvimg_n, racketUBox, 2)
            show_imgs(cvimg_c, cvimg_n, outimgs)

        #if (cnt >= 50):
        #    assert(False), "Temp forced exit to check work. Remove later."
            
        cnt += 1

cv.destroyAllWindows()
print("Done. Motion Dataset created with {} annotations and images".format(cnt))

