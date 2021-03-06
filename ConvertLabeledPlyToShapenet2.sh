#!/bin/bash
if [ $# -ne 2 ]
then
    echo "usage: startLocalWorker pathToLabeledPlyFolder pathToOutputFolder"
    exit 1
fi

for filename in $1/*.ply; do
    filename=${filename/$1/}
    ./build/pgm -J Shapenet2 --in $1 --out $2 --PointCloudName ${filename/.ply/}

done

exit 0