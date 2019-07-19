#!/usr/bin/env python
#
# Given a VOC dataset of TENNIS videos dumped at 1920x1080 resolution, this script creates a
# scaled and cropped dataset. Even though the cropped zone size is static (1280x720 at 2/3 
# crop scale), and the scaling is static with fixed scale of 0.5 (640x360), the zones 
# themselves are dynamically selected based on the objects locations (by reading the 
# annotations). 
# The zone size is selected for multiple reasons:
#   a. This size (2/3 of full scale) gives grid boxes of 1/3rd the full scale. This grid size
#      is the minimum overlap between the diagonal zones. Horizontal and vertically aligned
#      zones have the overlap that is double the height or width of this grid size. The 
#      minimum grid size is large enough to include a trail of tennis ball across three frames
#      even at fast speeds. This allows us to fully utilize motion information during training.
#   b. When images are cropped at 1280x720, and then finally scaled by 1/2, we get 640x360
#      as the final image size. This works perfectly with either 533x300 or 300x300 of final
#      training size while still allowing for random crop for training time data augmentation. 
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
from lxml import objectify
from glob import glob
import numpy as np
import subprocess
import re
import argparse
import copy

if sys.version_info[0] < 3:
    PYVER = 2
else:
    PYVER = 3

## INPUT IMAGE DIMENSIONS (scaled to these dimensions if required)
WIDTH  = 1920
HEIGHT = 1080

## MOTION DB setting: '3FRAMES' or 'FRAMESDIFF'
MOTION_TYPE = 'FRAMESDIFF'

## Change this to view images
SHOW_IMAGES = False

## Verbosity
DEBUG = 0


##############################################################################################
# Visualization of labels for debug

## Tennis Dataset labels
LBL_NAMES = [ "__bg__", "ball", "racket", "otherball", "player", "nonplayer" ]
LBL_COLORS = [
        (0xde, 0xde, 0xde), # background
        (0x48, 0x0c, 0xb8), # id 1
        (0x53, 0xb8, 0x09), # id 2
        (0xb8, 0x84, 0x0c), # ...
        (0x48, 0x0c, 0xb8),
        (0x53, 0xb8, 0x09),
        (0xb8, 0x84, 0x0c),
        (0x48, 0x0c, 0xb8),
        (0x53, 0xb8, 0x09),
        (0xb8, 0x84, 0x0c),
        (0x48, 0x0c, 0xb8),
        (0x48, 0x0c, 0xb8),
        (0x53, 0xb8, 0x09),
        (0xb8, 0x84, 0x0c),
        (0x48, 0x0c, 0xb8),
        (0x53, 0xb8, 0x09),
        (0xb8, 0x84, 0x0c),
        (0x53, 0xb8, 0x09),
        (0xb8, 0x84, 0x0c),
        (0x48, 0x0c, 0xb8),
        (0x53, 0xb8, 0x09)]


def writeBBLabelText(img, p1, p2, lblname, lblcolor, lblconf=None):
    '''
    img: image array
    p1, p2: bounding box upper-left and bottom-right points
    lblname: label name
    lblcolor: color tuple
    '''
    fontFace = cv.FONT_HERSHEY_PLAIN
    fontScale = 1.
    thickness = 1
    if lblconf is not None:
        lblname += ' {:3.0f}%'.format(lblconf*100)
    textSize, baseLine = cv.getTextSize(lblname, fontFace, fontScale, thickness)
    txtRBx = p1[0] + textSize[0] + 2
    if 0: ## Inside the box
        txtRBy = p1[1] + textSize[1] + 2
        img = cv.rectangle(img, p1, (txtRBx, txtRBy), lblcolor, cv.FILLED)
        textOrg = (p1[0]+thickness, p1[1]+textSize[1])
    else: ## Outside the box
        txtRBy = p1[1] - textSize[1] - 2
        img = cv.rectangle(img, p1, (txtRBx, txtRBy), lblcolor, cv.FILLED)
        textOrg = (p1[0]+thickness, p1[1]) #+textSize[1])
    img = cv.putText(img, lblname,
                     textOrg, fontFace, fontScale,
                     (255,255,255),      # inversecolor = Scalar::all(255)-lblcolor
                     thickness)
    return img


def drawBoundingBox(image, bbox, lblid, lblconf=None):
    """Draw bounding box on the image."""
    ymin,xmin,ymax,xmax = [int(f) for f in bbox]
    lblcolor = LBL_COLORS[lblid]
    lblname  = LBL_NAMES[lblid]
    image = cv.rectangle(image, (xmin, ymin), (xmax, ymax), lblcolor, 2)
    image = writeBBLabelText(image, (xmin, ymin), (xmax, ymax), lblname, lblcolor, lblconf)
    return image


def show_imgs(cvimg, cvimg_n, oimgs=[]):
    global SHOW_IMAGES
    cv.imshow("Original image", cvimg)
    cv.imshow("Motion image", cvimg_n)
    s = ["Out 1", "Out 2"]
    for i in range(len(oimgs)):
        cv.imshow(s[i], oimgs[i])
        
    key = cv.waitKey(0) & 255
    if key == 27:
        cv.destroyAllWindows()
        sys.exit(0)
    elif key == ord('g'): ## Go for it; don't show images after this
        cv.destroyAllWindows()
        SHOW_IMAGES = False


def getNumberingScheme(imgname):
    fnum     = re.sub(r'.*[-_](\d+).jpg', r'\1', imgname)
    fpre     = re.sub(r'(.*[-_])(\d+).jpg', r'\1', imgname)
    numlen   = len(fnum)
    numtmplt = '{:0' + str(numlen) + 'd}'
    return (fpre, numtmplt)


def getImageSizefromAnnotations(anndir, annfile):
    ann = getAnnotations(os.path.join(anndir, annfile))

    ## Get the image size from annotation
    return (ann.size.width, ann.size.height)


def getAnnotations(annfile):
    ''' Read XML annotations file, objectify and return
    '''
    with open(annfile) as f:
        xml = f.read()
    return objectify.fromstring(xml)


def scaleAnnotations(ann, SCALE):
    ## Change the size based on scale
    ann.size.width  = objectify.StringElement(str(int(ann.size.width  * SCALE)))
    ann.size.height = objectify.StringElement(str(int(ann.size.height * SCALE)))
    folder = ann.folder
    filename = ann.filename
    filepath = os.path.join(str(folder), 'JPEGImages', str(filename))
    ann.path = objectify.StringElement(filepath)
    for obj in ann.iter('object'):
        obj.bndbox.xmin = objectify.StringElement(str(obj.bndbox.xmin * SCALE))
        obj.bndbox.ymin = objectify.StringElement(str(obj.bndbox.ymin * SCALE))
        obj.bndbox.xmax = objectify.StringElement(str(obj.bndbox.xmax * SCALE))
        obj.bndbox.ymax = objectify.StringElement(str(obj.bndbox.ymax * SCALE))
    
    return ann

def cropAnnotations(_ann, roi, filename, minsize=4):
    '''
    For a given roi and annotation, returns a cropped annotation that contains all bounding
    boxes that can fit within the specified roi. If a bounding box is too small after
    cropping, it is discarded.
    Parameters:
    ann: objectified annotation xml
    roi: Input region of interest as [ymin,xmin,ymax,xmax]
    filename: New file name to use instead of the old one
    minsize: Resultant bounding boxes under this size (width or height) are discarded
    '''
    ymin,xmin,ymax,xmax = [int(f) for f in roi]

    ann = copy.deepcopy(_ann)
    ## Change the size based on RoI
    ann.size.width  = objectify.StringElement(str(xmax-xmin))
    ann.size.height = objectify.StringElement(str(ymax-ymin))
    folder = ann.folder
    ann.filename = objectify.StringElement(filename)
    filepath = os.path.join(str(folder), 'JPEGImages', str(filename))
    ann.path = objectify.StringElement(filepath)

    for obj in ann.iter('object'):
        b_xmin = obj.bndbox.xmin
        b_ymin = obj.bndbox.ymin
        b_xmax = obj.bndbox.xmax
        b_ymax = obj.bndbox.ymax
        # Clamp the bbox to RoI to make things simpler (this crops the bbox to remain within RoI)
        if b_xmin < xmin: b_xmin = xmin
        if b_ymin < ymin: b_xmin = ymin
        if b_xmax > xmax: b_xmax = xmax
        if b_ymax > ymax: b_xmax = ymax

        if (# T-L within RoI?
            (xmin <= b_xmin and b_xmin <= xmax and ymin <= b_ymin and b_ymin <= ymax) and
            # B-R within RoI?
            (xmin <= b_xmax and b_xmax <= xmax and ymin <= b_ymax and b_ymax <= ymax) and
            # minsize criteria met?
            ((b_xmax - b_xmin) >= minsize) and ((b_ymax - b_ymin) >= minsize)
            ):
            ## The [cropped] bbox is within the RoI. Now we only have to translate to (xmin,ymin) as origin
            obj.bndbox.xmin = objectify.StringElement(str(b_xmin-xmin))
            obj.bndbox.ymin = objectify.StringElement(str(b_ymin-ymin))
            obj.bndbox.xmax = objectify.StringElement(str(b_xmax-xmin))
            obj.bndbox.ymax = objectify.StringElement(str(b_ymax-ymin))
        else:
            ##--## Discard the obj --> Check this manually
            ##--obj_xml = etree.tostring(obj, pretty_print=True, xml_declaration=False)
            ##--print("Deleting object {}".format(obj_xml))
            obj.getparent().remove(obj)
    
    return ann

'''
A class to encapsulate bounding boxes, and realated operations
'''
class BoundingBoxes:

    def __init__(self, name):
        self.bboxes = []
        self.name = name

    def addBBox(self, bbox):
        box = [float(c) for c in bbox] ## Convert to float
        self.bboxes.append(box)

    def getBBox(self, boxid):
        return self.bboxes[boxid]

    def getImgRoI(self, boxid, img):
        '''
        Return an RoI from the input numpy.ndarray using selected boxid
        '''
        ymin,xmin,ymax,xmax = [int(f) for f in self.bboxes[boxid]]
        return img[ymin:ymax, xmin:xmax]

    def getNumBBoxes(self):
        return len(self.bboxes)

    def getUBox(self):
        '''
        Returns union of all bounding boxes
        '''
        if len(self.bboxes) == 0:
            return None
        ymin = min([b[0] for b in self.bboxes])
        xmin = min([b[1] for b in self.bboxes])
        ymax = max([b[2] for b in self.bboxes])
        xmax = max([b[3] for b in self.bboxes])
        ubox = [ymin, xmin, ymax, xmax]
        return ubox

    def findEnclosing(self, bbox):
        '''
        For a given bbox, finds a bbox that best encloses it
        The criteria for 'best' is the minimum distance from the center of the given bbox to
        bboxes stored in this class. A check is performed for the box with smallest distance
        to ensure that it does indeed enclose the given bbox. It the 'best' zone doesn't fully
        enclose the given bbox, a warning message is printed.
        Returns the index of the bbox that 'best' encloses the bbox.
        '''
        if bbox is None:
            return None
        # Find centers of bboxes in this class
        zcenters = [self.findCenter(b) for b in self.bboxes]
        bcenter = self.findCenter(bbox)
        distances = [cv.norm(bcenter, zcenter) for zcenter in zcenters]
        index = int(np.argmin(distances))
        if DEBUG>=2:
            print("Distances: {}".format(distances))
            print("Min distance: {}".format(distances[index]))
            print("Best Zone: {}".format(index))
        return index


    def findCenter(self, b):
        if DEBUG>=2:
            self.printBBox(b)
        ymin,xmin,ymax,xmax = b
        xc = xmin + (xmax-xmin)/2.0
        yc = ymin + (ymax-ymin)/2.0
        if DEBUG>=2:
            print("Center (x,y): ({:.2f}, {:.2f})".format(xc, yc))
        return (xc, yc)


    def printBBox(self,b):
        print("     ymin     xmin     ymax     xmax")
        str = ""
        for c in b:
            str += " {:8.2f}".format(float(c))
        print(str)


def getUBoxes(anns):
    '''
    Returns union of 'ball' and 'racket' bboxes.
    Will scale the annotations if not to the default global width
    Input anns: Array of objectified xml annotations
    '''
    ball_bboxes   = BoundingBoxes('ball')
    racket_bboxes = BoundingBoxes('racket')
    for ann in anns:
        for obj in ann.iter('object'):
            if obj.name == 'ball':
                ball_bboxes.addBBox([obj.bndbox.ymin, obj.bndbox.xmin, obj.bndbox.ymax, obj.bndbox.xmax])
            elif obj.name == 'racket':
                racket_bboxes.addBBox([obj.bndbox.ymin, obj.bndbox.xmin, obj.bndbox.ymax, obj.bndbox.xmax])

    return ball_bboxes.getUBox(), racket_bboxes.getUBox()
 
def drawZone(img, zid):
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
    return img


def runSystemCmd(cmd):
    '''
    Returns output as an array of lines. Error is discarded.
    '''
    # shell=True allows using a Unix pipe (for example) in the command
    proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    output,err = proc.communicate()
    if PYVER<3:
        output = output.rstrip().split('\n')
    else:
        output = (bytes.decode(output).rstrip()).split('\n')
    return output


def parseArgs():
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
        "--height", type=float, default=300, required=False,
        help="Output image height. "
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

## Find base datasets containing annotations
output = runSystemCmd(r"find {}/ -mindepth 3 -name '*.xml' | sed -e 's#/Annotations/.*.xml##g' | sort | uniq".format(IN_VOCDIR))
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
    for idir in [o_imgdir, o_anndir]:
        if not os.path.isdir(idir):
            os.makedirs(idir)
        else:
            print("Dir {} already exists".format(idir))

    ## Create image list to process
    imgs = glob("{}/*.jpg".format(i_imgdir))
    imgs = [os.path.basename(i) for i in imgs]
    imgs.sort() # Sort images to pick frames in order. It is assumed the images are named likewise

    (fprefix, ntemplate) = getNumberingScheme(imgs[0])
    #print("fprefix: {}, template: {}".format(fprefix, ntemplate))

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(4,4))
    
    ## Define the grid points
    gy = [0, int(HEIGHT/3.), int(HEIGHT*2.0/3.0), HEIGHT]
    gx = [0, int(WIDTH/3.), int(WIDTH*2.0/3.0), WIDTH]
    ## Create zones based on the grid
    zones = BoundingBoxes('zones')
    #              ymin   xmin   ymax   xmax
    zones.addBBox([gy[0], gx[0], gy[2], gx[2]])  # Top-left zone
    zones.addBBox([gy[0], gx[1], gy[2], gx[3]])  # Top-right zone
    zones.addBBox([gy[1], gx[0], gy[3], gx[2]])  # Bottom-left zone
    zones.addBBox([gy[1], gx[1], gy[3], gx[3]])  # Bottom-right zone

    annnames = glob("{}/*.xml".format(i_anndir))
    annnames = [os.path.basename(i) for i in annnames]
    annnames.sort() # Sort files to pick frames in order. It is assumed the xml/images are named likewise
    if len(annnames) < 3:
        print("This VOC Base has less than 3 annotations. Skipping.")
        continue

    i = 2  ## Index
    for ann in annnames[2:]:
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
            print("Skipping {}".format(img_i))
            continue
        else:
            if DEBUG>=1:
                print("Processing {}".format(img_i))

        ## Now that we got a three sequential frames, let's read annotations and get uboxes
        ## uboxes = union of bboxes for each of the 'ball' or 'racket' bbox in all three images
        ## We are assuming only one 'ball' annotation per image. However, it is easy to handle
        ## multiple balls per image too. Not needed for our dataset.
        annfiles = [fprefix + ntemplate.format(fn) + '.xml' for fn in [fnum, fnum-1, fnum-2]]
        anns = [getAnnotations(os.path.join(i_anndir, annfile)) for annfile in annfiles]
        w = anns[0].size.width #, h = getImageSizefromAnnotations(i_anndir, annfiles[0])

        ## Scale input to WIDTHxHEIGHT fixed dimensions if input size is different
        if w != WIDTH:
            scale = float(WIDTH) / float(w)
            ## Scale annotations
            anns = [scaleAnnotations(ann, scale) for ann in anns]
        else:
            scale = 1.0

        ballUBox, racketUBox = getUBoxes(anns)
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
            base = fprefix + ntemplate.format(fnum) + '-z{}'.format(zid)
            imgname = base + '.jpg'
            annname = base + '.xml'
            imgfilenames.append(imgname)
            annfilenames.append(annname)
            roi = zones.getBBox(zid)
            outann = cropAnnotations(anns[0], roi, imgname, 6)
            outimg = zones.getImgRoI(zid, cvimg_n).copy()
            outanns.append(outann)
            outimgs.append(outimg)
            if DEBUG>=3 and len(crop_zids) > 1:
                obj_xml = etree.tostring(outann, pretty_print=True, xml_declaration=False)
                print("Annotation {}\n{}".format(annname, obj_xml))


        ######################################################################################
        ## Write files 
        ######################################################################################
        for index in range(len(outimgs)):
            ## Write annotation files
            obj_xml = etree.tostring(outanns[index], pretty_print=True, xml_declaration=False)
            annfile = os.path.join(o_anndir, annfilenames[index])
            if DEBUG>=2:
                print("Writing {}".format(annfile))
            with open(annfile, 'w') as f:
                if PYVER>=3:
                    f.write(obj_xml.decode('utf8'))
                else:
                    f.write(obj_xml)

            ## Write cropped motion images
            imgfile = os.path.join(o_imgdir, imgfilenames[index])
            if DEBUG>=2:
                print("Writing {}".format(imgfile))
            cv.imwrite(imgfile, outimgs[index])

        if SHOW_IMAGES:
            for zid in crop_zids:
                cvimg_n = drawZone(cvimg_n, zid)
            ## Draw bounding boxes
            if ballUBox is not None:
                cvimg_n = drawBoundingBox(cvimg_n, ballUBox, 1)
            if racketUBox is not None:
                cvimg_n = drawBoundingBox(cvimg_n, racketUBox, 2)
            show_imgs(cvimg_c, cvimg_n, outimgs)

        #if (cnt >= 50):
        #    assert(False), "Temp forced exit to check work. Remove later."
            
        cnt += 1

cv.destroyAllWindows()
print("Done. Motion Dataset created with {} annotations and images".format(cnt))

