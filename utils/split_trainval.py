#! /usr/bin/env python3
#
#   File = split_trainval.py
#   Splits image lists to create VOC style train and val lists
#
# Run this as follows:
# split_trainval.py <VOCdevkitDir/VOCDatasetDIR>
# Where <VOCdevkitDir/VOCDatasetDIR> is the VOC dataset that contains Annotations
# directory. This script will create lists in ImageSets directory based on
# contents of Annotations directory.
############################################################################
import sys, os
from glob import glob
import random
import re


import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description='Split dataset into train, val and trainval lists in VOC format'
    )
    parser.add_argument(
        'vocdir', metavar='VOCDATASET_DIR',
        help='Dataset directory under VOCdevkit that contains Annotations, JPEGImages, etc.'
    )
    #parser.set_defaults(var=value)
    return parser.parse_args()



def splitDataset(vocdir):
    ## This will be created to save train.txt, val.txt, and trainval.txt
    imgsetsdir = os.path.join(vocdir, 'ImageSets', 'Main') 
    anndir = os.path.join(vocdir, 'Annotations')
    if not os.path.isdir(anndir):
        print("VOC dataset directory or Annotations within it do not exist")
        sys.exit(1)
    labels = [ os.path.splitext(os.path.basename(f.rstrip()))[0] for f in glob('{}/*.xml'.format(anndir)) ]
    random.shuffle(labels)
    print("Found {} labels".format(len(labels)))
    # 10% of labels for val
    nval = int(len(labels)*0.1)
    vlabels = labels[0:nval]
    tlabels = labels[nval:]

    print('Created {t} train and {v} val labels.'.format(v=len(vlabels), t=len(tlabels)))
    if not os.path.exists(imgsetsdir):
        os.makedirs(imgsetsdir)
    with open(os.path.join(imgsetsdir, 'train.txt'), 'w') as outFile:
        outFile.write('\n'.join(tlabels) + '\n')

    with open(os.path.join(imgsetsdir, 'val.txt'), 'w') as outFile:
        outFile.write('\n'.join(vlabels) + '\n')

    with open(os.path.join(imgsetsdir, 'trainval.txt'), 'w') as outFile:
        outFile.write('\n'.join(labels) + '\n')


def main():
    args = parse_args()
    splitDataset(args.vocdir)


if __name__ == "__main__":
    main()

