[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# SPRM - Spatial Process & Relationship Modeling
Ted Zhang, Haoran Chen, Matt Ruffalo, and Bob Murphy
Ray and Stephanie Lane Computational Biology Department
School of Computer Science
Carnegie Mellon University
V1.4.6 September 18, 2024

## Description
SPRM is a statistical modeling program that is used in the HuBMAP project to calculate a range of metrics, descriptors/features and models from multichannel tissue images.  It requires at a minimum a multichannel tissue image and a corresponding indexed image containing cell segmentations.  The metrics measure the quality of the provided images and the quality of the provided cell segmentation.  The descriptors are used for clustering of cells using various approaches and are saved for use in comparison with other tissue images and for content-based image search.  

SPRM can be used standalone with any type of multichannel 2D or 3D image (e.g., CODEX, IMS) for which a cell segmentation is available.

## Inputs

Primary: Two OMETIFF files -
1) a 3D or 4D multichannel intensity Image (2D or 3D for multiple channels)
2) a corresponding 3D or 4D Indexed Image containing one channel for each component of cell segmentation (currently “cells”, “nucleus”, “cell membrane” and “nuclear membrane”).

## Execution: (assuming SPRM.py is in working directory)
```bash
[python_path] SPRM.py --img-dir [img_dir_path] --mask-dir [mask_dir_path] --optional-img-dir [optional_img_dir_path] --output_dir [output_dir_path] --options_path [options_file_path] --celltype_labels [labels_file] --processes [number_of_processes_to_use]
```
## Outputs
OME-TIFFs showing pixel level results (remapping of channels) [3 per input image]
CSV containing interpolated cell outlines & polygons [2 per input image]
CSVs containing features for each cell [4 per image]
CSVs containing features for subcellular components segmentation [12 per image]
CSV containing clustering results for each cell (row) for different methods (column) [1 per input image]
CSVs containing mean values of “markers” for each cluster for each clustering method [5 per input image]
PNGs showing each cell colored by cluster for each clustering method [7 per input image]
CSV containing the signal to noise ratios of the image per channel [1 per input image]
CSV containing PCA and Silhouette analysis of the image [2 per input image]
JSON containing all features and cluster assignments

## Simple illustration

The demo folder contains two simple ways to run SPRM.  For both, begin by downloading the demo image files from [this link](https://drive.google.com/drive/folders/1denyZ1SFoWpWrPO9UbSdcF2DvHEv6ovN?usp=sharing).

* The shell script `run_sprm.sh` will just run SPRM and place the outputs in the folder sprm_demo_outputs and write a log of the messages from SPRM into the file sprm_demo_outputs/sprm_demo_outputs.log.  Run the command 
```bash
python ../setup.py install
```
beforehand.

*The jupyter notebook sprm_demo.ipynb will run `run_sprm.hs` on the example files and then display the outputs in the notebook. It will run setup.py inside the notebook.

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

[Documentation](https://docs.google.com/document/d/1aysD_yRmk_5Lmm2fXIUGCeWnxICpxrJt0Osym99FfWA/edit?usp=sharing)

## Development information

Code in this repository is formatted with [black](https://github.com/psf/black) and
[isort](https://pypi.org/project/isort/).

A [pre-commit](https://pre-commit.com/) hook configuration is provided, which runs `black` and `isort` before committing.
Run `pre-commit install` in each clone of this repository which you will use for development (after `pip install pre-commit`
into an appropriate Python environment, if necessary).

## Contact

Robert F. Murphy - murphy@cmu.edu\
Ted (Ce) Zhang - tedz@andrew.cmu.edu
