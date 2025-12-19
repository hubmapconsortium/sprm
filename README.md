[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# SPRM - Spatial Process & Relationship Modeling
Ted Zhang, Haoran Chen, Matt Ruffalo, and Robert F. Murphy\
Ray and Stephanie Lane Computational Biology Department\
School of Computer Science, Carnegie Mellon University\
V2.0.0 December 19, 2025

## Description
SPRM is a statistical modeling program that is used in the HuBMAP project to calculate a range of metrics, descriptors/features and models from multichannel tissue images.  It requires at a minimum a multichannel tissue image and a corresponding indexed image containing cell segmentations.  The metrics measure the quality of the provided images and the quality of the provided cell segmentation.  The descriptors are used for clustering of cells using various approaches and are saved for use in comparison with other tissue images and for content-based image search.  

SPRM can be used standalone with any type of multichannel 2D or 3D image (e.g., CODEX, IMC) for which a cell segmentation is available.

## Inputs

Primary: Two OMETIFF files -
1) a 3D or 4D multichannel intensity Image (2D or 3D for multiple channels)
2) a corresponding 3D or 4D Indexed Image containing one channel for each component of cell segmentation (“cells”, “nucleus”, “cell membrane” and “nuclear membrane”).

## Execution

### Option 1: Modular API (Recommended)

The modular API allows you to run individual components of SPRM independently.

#### Module Structure

SPRM is organized into 7 modules:

1. **Preprocessing** (Required) - Load images, extract ROIs, quality control
2. **Segmentation Evaluation** (Optional) - Assess segmentation quality
3. **Shape Analysis** (Optional) - Extract cell shape features
4. **Spatial Graphs (Cell Neighborhood Graphs)** (Optional) - Compute spatial relationships
5. **Image Analysis (Pixel Level Analysis)** (Optional) - NMF, superpixels, channel PCA
6. **Cell Features** (Required for clustering) - Intensity & texture features
7. **Clustering** (Analysis) - Multiple clustering methods

**Dependency Graph**

```
Preprocessing (Module 1) ✓ Always required
    ├─→ Segmentation Eval (Module 2) [standalone]
    ├─→ Shape Analysis (Module 3)
    ├─→ Spatial Graphs (Module 4)
    ├─→ Image Analysis (Module 5)
    └─→ Cell Features (Module 6) ✓ Required for clustering
         └─→ Clustering (Module 7)
```

**Critical path**: Preprocessing → Cell Features → Clustering  
**Optional branches**: Modules 2, 3, 4, 5 enhance analysis but aren't required

For example,

```python
from sprm import modules

# Run only what you need
core = modules.preprocessing.run(
    img_file="image.ome.tiff",
    mask_file="mask.ome.tiff",
    output_dir="sprm_outputs"
)

features = modules.cell_features.run(
    core_data=core,
    output_dir="sprm_outputs",
    compute_texture=False  # Skip texture for speed
)

clusters = modules.clustering.run(
    core_data=core,
    cell_features=features,
    output_dir="sprm_outputs"
)
```

**Benefits:**
- Run only the analyses you need
- Resume from checkpoints (save computation time)
- Re-run parts with different parameters
- Better for debugging and development

See [Modular API Documentation](docs/MODULAR_API.md) and [demo/](demo/) for more details.

### Option 2: Legacy Command-Line Interface

For backward compatibility, the original CLI still works:

```bash
[python_path] SPRM.py --img-dir [img_dir_path] --mask-dir [mask_dir_path] --optional-img-dir [optional_img_dir_path] --output_dir [output_dir_path] --options_path [options_file_path] --celltype_labels [labels_file] --processes [number_of_processes_to_use]
```
## Outputs
The following sets of files are placed in the folder specified by the “output_dir” argument.  They are

- OME-TIFFs showing pixel level results (remapping of channels) [3 per input image]
- CSV containing interpolated cell outlines & polygons [2 per input image]
- CSVs containing features for each cell [4 per image]
- CSVs containing features for subcellular components segmentation [12 per image]
- CSV containing clustering results for each cell (row) for different methods (column) [1 per input image]
- CSVs containing mean values of “markers” for each cluster for each clustering method [5 per input image]
- PNGs showing each cell colored by cluster for each clustering method [7 per input image]
- CSV containing the signal to noise ratios of the image per channel [1 per input image]
- CSV containing PCA and Silhouette analysis of the image [2 per input image]
- JSON containing all features and cluster assignments

The jupyter notebook “visualizeSPRMoutput.ipynb” can be used to visualize the important results files.

## Simple illustration

The demo folder contains two simple ways to run SPRM from the terminal. For both, begin by “cd”ing to the “demo” folder and downloading the demo image files from [this link](https://drive.google.com/drive/folders/1denyZ1SFoWpWrPO9UbSdcF2DvHEv6ovN?usp=sharing).

Beforehand, run the following command to install SPRM

```bash
pip install .
```

### Installing with pip


```bash
pip install sprmpkg
```

Even though the distribution name is `sprmpkg`, the Python import name remains:

```python
import sprm
```

Then you can either:

- Use the shell script `run_sprm.sh` which will just run SPRM using the downloaded demo image files. It will place the outputs in the folder sprm_demo_outputs and write a log of the messages from SPRM into the file sprm_demo_outputs/sprm_demo_outputs.log.
- Activate the SPRM environment (e.g., "conda activate SPRM") and then start jupyter notebook ("jupyter notebook"). Open the sprm_demo.ipynb notebook which will run SRPM on the demo files and then display the outputs in the notebook.

**EXAMPLES**

We have provided you with example images and masks but feel free to use your own as well! 

* downloading the demo image files from [this link](https://drive.google.com/drive/folders/1denyZ1SFoWpWrPO9UbSdcF2DvHEv6ovN?usp=sharing) and putting them in the "demo" folder into their own respective directories "img" and "mask".
## Prerequisites

* Python 3.8 or newer
* AICSImageIO
* h5py (for checkpoint system)
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

[Documentation](https://docs.google.com/document/d/1aysD_yRmk_5Lmm2fXIUGCeWnxICpxrJt0Osym99FfWA/view?usp=sharing)

## Development information

The notes below are provided in case you wish to make code changes and submit them to the SPRM github repository via a pull request.

Code in this repository is formatted with [black](https://github.com/psf/black) and
[isort](https://pypi.org/project/isort/).

A [pre-commit](https://pre-commit.com/) hook configuration is provided, which runs `black` and `isort` before committing.
Run `pre-commit install` in each clone of this repository which you will use for development (after `pip install pre-commit`
into an appropriate Python environment, if necessary).

## Contact

Robert F. Murphy - murphy@cmu.edu\
Ted (Ce) Zhang - tedz@andrew.cmu.edu\
Matt Ruffalo - mruffalo@cs.cmu.edu
