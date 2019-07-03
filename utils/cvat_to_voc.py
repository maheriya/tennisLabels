#!/usr/bin/env python3

## Make sure to install python modules as per opencv/cvat/utils/voc/requirements.txt
## before attempting this
##
## This script converts a CVAT generated annotations from XML file into a Pascal VOC
## dataset into the dir specified by --output-dir option.
## Since CVAT doesn't store images separately, it is a good idea to record the images
## directory while using CVAT. This is supplied as --image-dir option.
##
## This works best if you arrange your video dumps inside a VOC format directory
## structure before doing the annotations. That way, CVAT will use real paths
## in the XML, and after running this script, no further changes will be necessary
## This flow also sets things up nicely for conversion to TFRecord format if you need
## to go that way.
import os
import sys
import hashlib
import argparse


ANNOTATIONS_DIR_ = "/IMAGESETS/TENNIS/tennisLabels/annotations"
IMGROOTDIR_      = "/IMAGESETS/TENNIS/VOCdevkit"
## Do we need to create a hash for directory name?
CREATE_HASH_     = False




def cvat_to_voc(img_n_xml_base, annotations_dir, IMGROOTDIR, CREATE_HASH):

    if CREATE_HASH:
        ## images base name
        imgsH = img_n_xml_base.split('/')[0]
        ## Create hash to avoid very long names
        hname = imgsH + "-" + hashlib.md5(str.encode(img_n_xml_base)).hexdigest()[0:8] 
    else:
        hname = img_n_xml_base
    
    ## Input CVAT XML file
    annxml = os.path.join(annotations_dir, hname + ".xml")
    ## Path to VOC annotations
    outdir = os.path.join("/IMAGESETS/TENNIS/VOCdevkit", hname, "Annotations")
    ## Path to images
    imgdir = os.path.join(IMGROOTDIR, hname, "JPEGImages")
    print("Input CVAT XML     : {}".format(annxml))
    print("Original images    : {}".format(imgdir))
    print("Output Annotations : {}".format(outdir))
 
    sys.path.insert(0, os.path.join(os.environ['HOME'], "Projects/opencv_cvat/utils"))
    from voc.converter import process_cvat_xml
    
    ## Run the converter
    process_cvat_xml(annxml, imgdir, outdir)

def parse_args():
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(
        description='Convert CVAT XML annotations to PASCAL VOC format'
    )
    parser.add_argument(
        '--annotations-dir', metavar='DIR', required=False, default=ANNOTATIONS_DIR_,
        help='Directory where annotations XML are stored'
    )
    parser.add_argument(
        '--imgs-root-dir', metavar='DIR', required=False, default=IMGROOTDIR_,
        help='Directory where original images are stored'
    )
    parser.add_argument(
        '--no-hash', dest='dohash', action='store_false', required=False,
        help='Do not create a hash from xml_or_img_basename.'
    )
    parser.add_argument(
        '--hash', dest='dohash', action='store_true', required=False,
        help='Create a hash from xml_or_img_basename to shorten length of path'
    )
    parser.add_argument(
        'xml_or_img_basename', metavar='IMG_n_XML_BASE',
        help='XML name without .xml (which is also used as base directory path for original images)'
    )
    parser.set_defaults(dohash=CREATE_HASH_)
    return parser.parse_args()



    
def main():
    args = parse_args()
    ##--print("annotation-dir: {}".format(args.annotations_dir))
    ##--print("imgs-root-dir: {}".format(args.imgs_root_dir))
    ##--print("dohash: {}".format(args.dohash))
    ##--print("img_n_xml_base: {}".format(args.xml_or_img_basename))
    cvat_to_voc(args.xml_or_img_basename, args.annotations_dir, args.imgs_root_dir, args.dohash)

if __name__ == "__main__":
    main()

