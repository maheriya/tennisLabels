'''
Created on Jul 19, 2019

@author: maheriya
'''

from __future__ import print_function
import os
import sys
import cv2 as cv

from lxml import objectify
from lxml import etree

import numpy as np
import subprocess
import re
import copy

if sys.version_info[0] < 3:
    PYVER = 2
else:
    PYVER = 3

DEBUG = 0
## Tennis Dataset labels
LBL_NAMES = [ "__bg__", "ball", "racket", "pballs", "player", "nonplayer" ]
LBL_IDS = { 
        LBL_NAMES[0] : 0,
        LBL_NAMES[1] : 1,
        LBL_NAMES[2] : 2,
        LBL_NAMES[3] : 3,
        LBL_NAMES[4] : 4,
        LBL_NAMES[5] : 5}
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
    image = cv.rectangle(image, (xmin, ymin), (xmax, ymax), lblcolor, 1)
    image = writeBBLabelText(image, (xmin, ymin), (xmax, ymax), lblname, lblcolor, lblconf)
    return image


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

def getBboxfromAnnotation(ann, lbl):
    ''' From the given objectified xml ann, return bbox for first label lbl
    '''
    bbox = None
    for obj in ann.iter('object'):
        if obj.name == lbl:
            bbox = [obj.bndbox.ymin, obj.bndbox.xmin, obj.bndbox.ymax, obj.bndbox.xmax]
            break
    return bbox


def addAnnotation(ann, lbl, bbox):
    '''Adds a label 'lbl' with bounding box 'bbox' in the annotation xml ann
    '''
    obj = objectify.SubElement(ann, 'object')
    obj.name = lbl
    obj.pose = 'Unspecified'
    obj.truncated = 0
    obj.difficult = 0
    bndbox = objectify.SubElement(obj, 'bndbox')
    bndbox.ymin, bndbox.xmin, bndbox.ymax, bndbox.xmax = bbox

def cleanUpAnnotations(_ann, lbls):
    '''
    For a given annotation, returns a cleaned up annotation only contains specified labels as
    defined in the array 'lbls'
    Also removes pesty names spaces or py types that might get added.
    '''
    objectify.deannotate(_ann, xsi_nil=True) ## Remove those py type annotations
    etree.cleanup_namespaces(_ann)           ## Remove namespaces
    for obj in _ann.iter('object'):
        if obj.name not in lbls:
            ## Discard/delete the obj from annotation xml
            obj.getparent().remove(obj)
    return _ann


def writeAnnotation(ann, annfile):
    '''
    Takes as input the objectified xml ann, and writes into the specified file annfile
    '''
    obj_xml = etree.tostring(ann, pretty_print=True, xml_declaration=False)
    if DEBUG>=2:
        print("Writing {}".format(annfile))
    with open(annfile, 'w') as f:
        if PYVER>=3:
            f.write(obj_xml.decode('utf8'))
        else:
            f.write(obj_xml)

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
A class to encapsulate bounding boxes, and related operations
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


