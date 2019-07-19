#!/bin/csh -f

if ($#argv != 3) then
  echo "Usage: convert_all_to_voc.csh <tennisLabelsRepoDir> <cvat_annoations_dir> <VOCdevkit_dir>"
  exit(1)
endif

set repodir = $argv[1]
set anndir = $argv[2]
set vocdir = $argv[3]

echo "Picking all annotations from $anndir"
echo "VOC annotations will be extracted into $vocdir"
set cdir = `pwd`

foreach xml ( `cd $anndir && /bin/ls *.xml` )
  set base = `echo $xml | sed 's#.xml##g'`
  echo $base
  $repodir/utils/cvat_to_voc.py --annotations-dir=$anndir --imgs-root-dir=$vocdir $base
end
