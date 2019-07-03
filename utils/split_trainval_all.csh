#!/bin/csh -f
#

if ($#argv != 2) then
  echo "Usage: split_trainval_all.csh <tennisLabelsRepoDir> <VOCdevkit_dir>"
  exit(1)
endif

set cdir = `pwd`
set repodir = `cd $argv[1] && pwd`
set anndir = "$repodir/annotations"
set vocdir = `cd $argv[2] && pwd `

echo "Picking all datasets from $anndir"
echo "Splitting VOC datasets in $vocdir"
set bases = ( `cd $anndir && /bin/ls *.xml | sed 's#.xml##g'` )
cd $vocdir
echo "In $vocdir"
foreach base ( $bases )
  echo $base
  @ cnt = `find $vocdir/$base/Annotations/ -name '*.xml' | wc -l`
  if ( $cnt < 2 ) then
    ## Skip if not enough annotations
    continue
  endif
  #which $repodir/utils/split_trainval.py
  $repodir/utils/split_trainval.py $base
end

