#!/bin/csh -f

if ($#argv != 2) then
  echo "Usage: convert_all_to_voc.csh <tennisLabelsRepoDir> <VOCdevkit_dir>"
  exit(1)
endif

set repodir = $argv[1]
set anndir = $repodir/annotations
set vocdir = $argv[2]

echo "Picking all annotations from $anndir"
echo "Extracting VOC annotations into $vocdir"
set cdir = `pwd`
set bases = ( `cd $anndir && /bin/ls *.xml | sed 's#.xml##g'` )

foreach base ( $bases )
  echo $base
  $repodir/utils/cvat_to_voc.py $base
end

