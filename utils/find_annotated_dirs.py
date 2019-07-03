#!/usr/bin/env python
#
# Simple utility to find all VOCdevkit/Annotations directories that
# contain annotations (at least one annotation xml file)
import os
import sys
from glob import glob
import subprocess

IN_VOCDIR = "VOCdevkit"
if (len(sys.argv)==2):
    IN_VOCDIR = sys.argv[1]
findtask = subprocess.Popen(
    [r"find {}/ -mindepth 3 -name '*.xml' | sed -e 's#/Annotations/.*.xml##g' | sort | uniq".format(IN_VOCDIR)], 
     shell=True, stdout=subprocess.PIPE)
output,err = findtask.communicate()
if sys.version_info[0] < 3:
    output = output.rstrip().split('\n')
else:
    output = (bytes.decode(output).rstrip()).split('\n')
print("Annotated datasets under {}:".format(IN_VOCDIR))
vocbases = [os.path.basename(d) for d in output]
for base in vocbases:
    print(base)
print("There are {} datasets with annotations".format(len(vocbases)))
