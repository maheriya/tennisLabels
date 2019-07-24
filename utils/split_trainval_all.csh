#!/bin/csh -f
#

if ($#argv != 2) then
  echo "Usage: split_trainval_all.csh <tennisLabelsRepoDir> <VOCdevkit_dir>"
  exit(1)
endif

set cdir = `pwd`
set repodir = $argv[1]
set vocdir = $argv[2]
if (! -d $repodir) then
  echo "$repodir doesn't exist"
  exit(1)
endif
if (! -d $vocdir) then
  echo "$vocdir doesn't exist"
  exit(1)
endif
set repodir = `cd $repodir && pwd`
set vocdir = `cd $vocdir && pwd `
echo "Splitting VOC datasets in $vocdir"

cd $vocdir || exit(1)
echo "In $vocdir"
foreach base ( `/bin/ls` )
  @ cnt = `find $vocdir/$base/Annotations/ -name '*.xml' | wc -l`
  if ( $cnt < 2 ) then
    ## Skip if not enough annotations
    continue
  endif
  echo $base
  #which $repodir/utils/split_trainval.py
  $repodir/utils/split_trainval.py $base
end

