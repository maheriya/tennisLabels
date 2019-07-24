#!/bin/csh -f

## This is a script that combines multiple voc directories into a smaller number of directories
## There is no check for name conflicts. Files with the same names will be overwritten.
## The VOC database should be created with uniq names if you use this script.
if ($#argv != 2) then
  echo "Usage: $0 <invoc> <outvoc>"
  exit(1)
endif

set invoc = `cd $argv[1] && pwd`
set outvoc = $argv[2]
if (-d $outvoc) then
  echo "Output VOC already exists"
  #exit(1)
else
  mkdir -p $outvoc
endif
set outvoc = `cd $outvoc && pwd`
echo "Input VOC: $invoc"
echo "Output combined VOC: $outvoc"
set vocbases = ( `cd $invoc ; find . -maxdepth 2 -mindepth 2 -type d -name 'JPEGImages' | sed -e 's#^\./##' -e 's#/JPEGImages##'`)
@ nvoc = `echo $vocbases | wc -w`
@ ndirs = $nvoc / 12 ## Every 12 directories combined into one
@ m = $nvoc % 12
if ( $m != 0) @ ndirs = $ndirs + 1
echo "Number of output directories: $ndirs"

@ cnt = 0
@ vcnt = 0
while ($cnt < $ndirs)
  @ cnt = $cnt + 1
  set odir = "set$cnt"
  echo "Combining into $odir"
  if (! -d $outvoc/$odir) mkdir -p $outvoc/$odir
  @ d = 0
  while ($d < 12 && $vcnt < $nvoc)
    @ d = $d + 1
    @ vcnt = $vcnt + 1
    set vocbase = $vocbases[$vcnt]
    echo "Copying $vcnt / $nvoc ($vocbase)"
    cp -r $invoc/$vocbase/JPEGImages/ $outvoc/$odir/
    set oanndir = $outvoc/$odir/Annotations
    if (! -d $oanndir) mkdir -p $oanndir
    foreach xml ( `cd $invoc/$vocbase/Annotations && /bin/ls *.xml` )
      cat $invoc/$vocbase/Annotations/$xml | sed "s#$vocbase#$odir#" > $oanndir/$xml
    end
  end
end
