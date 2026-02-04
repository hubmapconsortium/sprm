[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# SPRM - Spatial Process & Relationship Modeling
Ted Zhang, Haoran Chen, Matt Ruffalo, and Robert F. Murphy\
Ray and Stephanie Lane Computational Biology Department\
School of Computer Science, Carnegie Mellon University\
	V2.0.3 December 23, 2025

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
    options="sprm/options.txt"  # Pass path to options file
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

NOTE: Intermediate results are saved in the "checkpoints" folder to allow continuing analysis if it is interrupted.  These files in this folder can be very large and can be deleted when analysis is finished.

### Option 2: Legacy Command-Line Interface

For backward compatibility, the original CLI still works:

```bash
[python_path] SPRM.py --img-dir [img_dir_path] --mask-dir [mask_dir_path] --optional-img-dir [optional_img_dir_path] --output-dir [output_dir_path] --options-file [options_file_path] --celltype-labels [labels_file] --processes [number_of_processes_to_use]
```

#### Memory usage (`--min-memory`)

If you are running out of RAM, you can pass `--min-memory` to **keep large image arrays on disk where possible** (SPRM will stream planes from the OME-TIFF instead of loading the full image into memory).

```bash
[python_path] SPRM.py --img-dir [img_dir_path] --mask-dir [mask_dir_path] --output-dir [output_dir_path] --options-file [options_file_path] --processes 4 --min-memory
```

- **Tradeoff**: lower peak RAM, but typically slower due to increased disk I/O.
- **Important limitations**:
  - Image “bias” normalization for negative-valued pixels is **not supported** in min-memory mode (inputs should already be non-negative / normalized).
  - Segmentation evaluation metrics in min-memory mode currently require a **single Z slice** (i.e., effectively 2D / `Z==1`, and only one `bestz`).
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

The demo folder contains two simple ways to run SPRM from the terminal. For both, begin by installing SPRM, either using the install script from this repo or by manually creating an environment containing it using the `sprmpkg` package from PyPI

```bash
bash install_sprm.sh
```

or


```bash
conda create -name SPRM python==3.11 -y
conda activate SPRM
conda install numba
pip install sprmpkg
```

Even though the PyPI name is `sprmpkg`, the Python import name remains `sprm` (i.e., load using `import sprm`).

Then download the demo image files from [this link](https://drive.google.com/drive/folders/1denyZ1SFoWpWrPO9UbSdcF2DvHEv6ovN?usp=sharing) into the img and mask folders

```
bash downloaddemofiles.sh
```

Then you can use one of the following approaches:

- Use the shell script `run_sprm.sh` which will just run SPRM using the downloaded demo image files. It will place the outputs in the folder sprm_demo_outputs and write a log of the messages from SPRM into the file sprm_demo_outputs/sprm_demo_outputs.log.  You can use “visualizeSPRMoutput.ipynb” to display the important results.
- Start jupyter notebook ("jupyter notebook") and open the sprm_demo.ipynb notebook which will run SRPM on the demo files and then display the outputs in the notebook.
- Use one of the python scripts (demo*.py) that use the individual SPRM modules.  Activate the SPRM environment (e.g., "conda activate SPRM") and run an example script (e.g., `python demo_features_only.py').  The scripts will put the outputs into the "sprm_demo_outputs" folder.  See NOTE above about large files in the "checkpoints" folder that can be deleted after processing.

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
