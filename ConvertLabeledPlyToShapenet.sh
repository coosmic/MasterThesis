#!/bin/bash
if [ $# -ne 2 ]
then
    echo "usage: ConvertLabeledPlyToShapenet pathToLabeledPlyFolder pathToOutputFolder"
    exit 1
fi

for filename in $1/*.ply; do
    filename=${filename/$1/}
    #./build/pgm $1/$filename $2/${filename/.ply/.txt}
    #./build/pgm -J Shapenet --snin $1/$filename --snout $2/${filename/.ply/.txt}
    ./build/pgm -J Shapenet --snin $1/$filename --snout $2${filename/.ply/} --RemoveBackground false

done

exit 0