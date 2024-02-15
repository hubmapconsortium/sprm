# Introduction 

SPRM is a statistical modeling program used to calculate a range of descriptors/features from multichannel cell images.  It uses these descriptors for clustering of cells and saves them for use for comparison and content-based search of the images.  It can be used with any type of multichannel 2D or 3D image (e.g., CODEX, IMS).

### Inputs

Two OMETIFF files from the output of CytoKit -
4D Multiplexed intensity Image (3D for multiple channels)
4D Indexed Image (3D for multiple segmentations) containing one channel for each type of segmentation (currently “cells”, “nucleus”, “cell membrane” and “nuclear membrane”).

SPRM takes in the following command line arguments to specify required and optional inputs.
```commandline
SPRM.py [img_dir_path] [mask_dir_path] [output_dir_path] [options_path] [cell-type]
```
## Requirements

* aicsimageio
* numpy
* sklearn
* pandas
* matplotlib
* scipy
* pillow
* python-cv2
* numba
* shapely

## Contact 

For any inquires and question please contact us!

Robert F. Murphy \
Ted Zhang \
Haoran Chen 


