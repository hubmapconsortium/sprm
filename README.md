# SPRM - Spatial Process & Relationship Modeling
Ted Zhang and Bob Murphy, Carnegie Mellon University
V1.0 March 1, 2021

## Description
SPRM is a statistical modeling program used to calculate a range of descriptors/features from multichannel cell images.  It uses these descriptors for clustering of cells and saves them for use for comparison and content-based search of the images.  It can be used with any type of multichannel 2D or 3D image (e.g., CODEX, IMS).

## Inputs

Two OMETIFF files from the output of CytoKit -
4D Multiplexed intensity Image (3D for multiple channels)
4D Indexed Image (3D for multiple segmentations) containing one channel for each type of segmentation (currently “cells”, “nucleus”, “cell membrane” and “nuclear membrane”).

## Execution: (assuming SPRM.py is in working directory)
```bash
[python_path] SPRM.py [img_dir_path] [mask_dir_path] [options_img_dir_path]
```

SPRM takes in three command line arguments that specify the path in the following order:
Image directory path\
Mask directory path\
Options image file path

## Prerequisites

Aicsimageio\
Numpy\
Sklearn\
Pandas\
Matplotlib\
Numba\
Scipy\
PIL

## Documentation 

For more information on specific analytical tools and outputs of SPRM: 

[Documentation](https://docs.google.com/document/d/1ZSH9Ek8C4Ucvaytwxyg8LdrgLU6EHmgAsJ5z7Tcc8HQ/edit#heading=h.5y17kqj4hpjb)

## Contact

Robert F. Murphy - murphy@cmu.edu\
Ted (Ce) Zhang - tedz@andrew.cmu.edu
