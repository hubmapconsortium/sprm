#!/bin/bash

# this script assumes it is run from the demo folder

# Path to the sprm executable
SPRM_PATH="../SPRM.py"

demodir=`pwd`
#echo 'Running from '$demodir
if [ ! -e "sprm_demo_outputs" ]; then
    mkdir "sprm_demo_outputs"
fi

#python -u $SPRM_PATH --img-dir img --mask-dir mask --output-dir sprm_demo_outputs > sprm_demo_outputs/
python -u $SPRM_PATH --img-dir $demodir/image_demo.ome.tiff --mask-dir $demodir/mask_demo.ome.tiff --output-dir $demodir/sprm_demo_outputs > $demodir/sprm_demo_outputs/sprm_demo_outputs.log
echo "Finished running sprm"
