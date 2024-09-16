#!/bin/bash

# Path to the sprm executable
SPRM_PATH="../SPRM.py"

# Command to run sprm
python -u $SPRM_PATH --img-dir img --mask-dir mask --output-dir sprm_demo_outputs > sprm_demo_outputs/sprm_demo_outputs.log

# print out finish when done
echo "Finish running sprm"