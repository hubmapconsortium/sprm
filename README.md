[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# SPRM - Spatial Process & Relationship Modeling
Ted Zhang and Bob Murphy, Carnegie Mellon University
V1.40 January 6th, 2024

## Description
SPRM is a statistical modeling program used to calculate a range of descriptors/features from multichannel cell images.  It uses these descriptors for clustering of cells and saves them for use for comparison and content-based search of the images.  It can be used with any type of multichannel 2D or 3D image (e.g., CODEX, IMS).

## Inputs

Two OMETIFF files from the output of CytoKit -
4D Multiplexed intensity Image (3D for multiple channels)
4D Indexed Image (3D for multiple segmentations) containing one channel for each type of segmentation (currently “cells”, “nucleus”, “cell membrane” and “nuclear membrane”).

## Execution: (assuming SPRM.py is in working directory)
```bash
[python_path] SPRM.py --img-dir [img_dir_path] --mask-dir [mask_dir_path] --optional-img-dir [optional_img_dir_path]
```

SPRM takes in three command line arguments that specify the path in the following order:\
Image directory path\
Mask directory path\
Options image file path

## Prerequisites

* Python 3.8 or newer
* AICSImageIO
* Matplotlib
* Numba
* NumPy
* Pandas
* Pillow
* Pint
* scikit-learn
* SciPy
* Shapely

## Documentation 

For more information on specific analytical tools and outputs of SPRM: 

[Documentation](https://docs.google.com/document/d/1ZSH9Ek8C4Ucvaytwxyg8LdrgLU6EHmgAsJ5z7Tcc8HQ/edit#heading=h.5y17kqj4hpjb)

## Development information

Code in this repository is formatted with [black](https://github.com/psf/black) and
[isort](https://pypi.org/project/isort/).

A [pre-commit](https://pre-commit.com/) hook configuration is provided, which runs `black` and `isort` before committing.
Run `pre-commit install` in each clone of this repository which you will use for development (after `pip install pre-commit`
into an appropriate Python environment, if necessary).

## Contact

Robert F. Murphy - murphy@cmu.edu\
Ted (Ce) Zhang - tedz@andrew.cmu.edu
