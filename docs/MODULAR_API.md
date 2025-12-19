# SPRM Modular API Documentation

## Overview

The SPRM modular API allows you to run individual components of the SPRM pipeline independently. This enables:

- **Selective execution**: Run only the analyses you need
- **Checkpoint/resume**: Save computation time by resuming from saved checkpoints
- **Parameter exploration**: Re-run analyses with different parameters without recomputing everything
- **Memory efficiency**: Load only the data you need
- **Better debugging**: Isolate and test individual components

## Architecture

### Module Structure

SPRM is organized into 7 modules:

1. **Preprocessing** (Required) - Load images, extract ROIs, quality control
2. **Segmentation_Eval** (Optional) - Assess segmentation quality
3. **Shape_Analysis** (Optional) - Extract cell shape features
4. **Spatial_Graphs (Cell Neighborhood Graphs)** (Optional) - Compute spatial relationships
5. **Image_Analysis (Pixel Level Analysis)** (Optional) - NMF, superpixels, channel PCA
6. **Cell_Features** (Required for clustering) - Intensity & texture features
7. **Clustering** (Analysis) - Multiple clustering methods

### Dependency Graph

```
Preprocessing (Module 1) ✓ Always required
    ├─→ Segmentation Eval (Module 2) [standalone]
    ├─→ Shape Analysis (Module 3)
    ├─→ Spatial Graphs (Module 4a)
    ├─→ Image Analysis (Module 4b)
    └─→ Cell Features (Module 5) ✓ Required for clustering
         └─→ Clustering (Module 6)
```

**Critical path**: Preprocessing → Cell Features → Clustering  
**Optional branches**: Modules 2, 3, 4a, 4b enhance analysis but aren't required

## Module Reference

### Module 1: Preprocessing

**Purpose**: Load images, extract cell ROI coordinates, perform quality control, create data object to hold results

**Required for**: All other modules

```python
from sprm import modules

# Basic usage with defaults
core = modules.preprocessing.run(
    img_file="image.ome.tiff",
    mask_file="mask.ome.tiff",
    output_dir="sprm_outputs",
    image_dimension="2D",  # or "3D"
    debug=False
)

# With options file (legacy compatibility)
core = modules.preprocessing.run(
    img_file="image.ome.tiff",
    mask_file="mask.ome.tiff",
    output_dir="sprm_outputs",
    options="sprm/options.txt"  # Path to options file
)

# With options dictionary
my_options = {
    "image_dimension": "2D",
    "debug": 1,
    "interior_cells_only": 1,
    "valid_cell_threshold": 10,
    # ... other options
}
core = modules.preprocessing.run(
    img_file="image.ome.tiff",
    mask_file="mask.ome.tiff",
    output_dir="sprm_outputs",
    options=my_options
)
```

**Parameters**:
- `img_file`: Path to OME-TIFF image
- `mask_file`: Path to OME-TIFF mask/segmentation
- `output_dir`: Output directory for results and checkpoints
- `options` (optional): Path to options.txt, dict of options, or None (use defaults). See [Main Documentation](https://docs.google.com/document/d/1aysD_yRmk_5Lmm2fXIUGCeWnxICpxrJt0Osym99FfWA/view?usp=sharing) for list of options.
- `image_dimension`: "2D" or "3D" (used if options is None)
- `debug`: Enable debug output (used if options is None)

**Returns**: `CoreData` object containing:
- `im`: IMGstruct (image data)
- `mask`: MaskStruct (segmentation data)
- `roi_coords`: Cell coordinate lists
- `interior_cells`, `edge_cells`, `cell_index`: Cell lists
- `bad_cells`: Set of invalid cell indices
- `bestz`: Best focal plane(s)
- `cell_centers`: Array of per-cell centroid coordinates (x, y, z)

**Checkpoints saved**:
- `checkpoints/core_data.pkl`
- `checkpoints/roi_coords.h5`
- `checkpoints/cell_lists.json`

**Outputs**:
- `{image_name}-cell_centers.csv`

---

### Module 2: Segmentation Evaluation

**Purpose**: Compute segmentation quality metrics

**Depends on**: Module 1

```python
seg_metrics = modules.segmentation_eval.run(
    core_data=core,  # or path to checkpoint directory
    output_dir="sprm_outputs",
    image_dimension="2D",
    standalone=False  # Set True to exit after evaluation
)
```

**Returns**: `SegmentationMetrics` with quality measures

**Outputs**:
- `{image_name}-SPRM_Image_Quality_Measures.json`

---

### Module 3: Shape Analysis

**Purpose**: Extract cell shape features using parametric outlines and PCA

**Depends on**: Module 1

```python
shape = modules.shape_analysis.run(
    core_data=core,
    output_dir="sprm_outputs",
    n_outline_points=100,
    debug=False  # Set True for PCA visualizations
)
```

**Returns**: `ShapeData` containing:
- `shape_vectors`: Raw shape features
- `norm_shape_vectors`: Normalized shape features
- `outline_vectors`: Parametric outlines
- `cell_polygons`: Cell polygon representations

**Checkpoints saved**:
- `checkpoints/shape_features.h5`
- `checkpoints/cell_polygons.pkl`

**Outputs**:
- `{image_name}-cell_outline.csv`
- `{image_name}-cell_polygons.csv`

---

### Module 4a: Spatial Graphs

**Purpose**: Compute spatial relationships between cells

**Depends on**: Module 1

```python
spatial = modules.spatial_graphs.run(
    core_data=core,
    output_dir="sprm_outputs",
    adjacency_dilation_itr=3,
    adjacency_delta=3,
    compute_cell_graph=True
)
```

**Returns**: `SpatialData` containing:
- `adjacency_matrix`: Cell adjacency relationships
- `cell_graph`: Detailed graph structure
- `snr_data`: Signal-to-noise ratio per channel

**Checkpoints saved**:
- `checkpoints/spatial_data.pkl`

**Outputs**:
- `{image_name}-adjacency_matrix.csv`
- `{image_name}-cell_graph.pkl`
- `{image_name}-SNR.csv`

---

### Module 4b: Image Analysis

**Purpose**: Whole-image analyses (NMF, superpixels, channel PCA)

**Depends on**: Module 1

```python
image_analysis = modules.image_analysis.run(
    core_data=core,
    output_dir="sprm_outputs",
    n_voxel_clusters=3,
    n_channel_pca_components=3,
    compute_nmf=True,
    generate_visualization_tiff=True
)
```

**Returns**: `ImageAnalysisData` containing:
- `nmf_results`: NMF decomposition results
- `superpixels`: Pixel cluster image
- `pca_components`: Channel PCA components
- `pca_img`: PCA visualization image

**Checkpoints saved**:
- `checkpoints/image_analysis.pkl`

**Outputs**:
- `{image_name}-Superpixels.png`
- `{image_name}-Top3ChannelPCA.png`
- `{image_name}-PCA_silhouette.csv`
- `{image_name}-NMF_*.csv`
- `{image_name}-visualization.ome.tiff`

---

### Module 5: Cell Features

**Purpose**: Extract per-cell intensity statistics and texture features

**Depends on**: Module 1, optionally Module 3

```python
features = modules.cell_features.run(
    core_data=core,
    shape_data=shape,  # Optional
    output_dir="sprm_outputs",
    optional_img_file=None,  # Additional image to merge
    compute_texture=True,  # Set False to skip (faster)
    glcm_angles=[0],
    glcm_distances=[1]
)
```

**Returns**: `CellFeatures` containing:
- `mean_vector`: Mean intensities per cell
- `covar_matrix`: Covariance matrices per cell
- `total_vector`: Total intensities per cell
- `texture_vectors`: GLCM texture features
- `texture_channels`: Texture channel names

**Checkpoints saved**:
- `checkpoints/cell_features.h5`

**Outputs**:
- `{image_name}-cell_channel_meanAll.csv`
- `{image_name}-{channel}_channel_mean.csv`
- `{image_name}-{channel}_channel_covar.csv`
- `{image_name}-{channel}_channel_total.csv`
- `{image_name}-{channel}_texture.csv`

---

### Module 6: Clustering

**Purpose**: Perform clustering and cell type analysis

**Depends on**: Modules 1, 5; optionally 3, 4a, 4b

```python
clusters = modules.clustering.run(
    core_data=core,
    cell_features=features,
    shape_data=shape,  # Optional
    spatial_data=spatial,  # Optional
    image_analysis_data=image_analysis,  # Optional
    output_dir="sprm_outputs",
    celltype_labels=None,  # Path to CSV with labels
    n_clusters_range=(3, 10),
    n_clusters_step=1,
    clustering_method="silhouette",  # or "fixed"
    n_shape_clusters_range=(3, 6)
)
```

**Clustering methods applied**:
- K-means on mean intensities
- K-means on covariance matrices
- K-means on total intensities
- K-means on texture features
- K-means on shape features (if available)
- t-SNE + dimensionality reduction + clustering
- UMAP + dimensionality reduction + clustering

**Returns**: `ClusteringResults`

**Outputs**:
- `{image_name}-{channel}_clusterIDs.csv`
- `{image_name}-Legend_{method}.csv` (cluster centers/markers)
- `{image_name}-{method}_clusters.png` (visualizations)
- `{image_name}-all_features.json`
- `{image_name}-clustering_scores.csv`

---

## Checkpoint System

### How It Works

Each module saves its outputs as checkpoints in `{output_dir}/checkpoints/`:

- **Automatic**: Checkpoints saved automatically after each module completes
- **Smart loading**: If checkpoint exists, module loads it instead of recomputing
- **Resume-friendly**: Interrupted runs can resume from last checkpoint
- **Path-based**: Can pass directory path instead of data object

### Using Checkpoints

**Option 1: Pass data objects**
```python
core = modules.preprocessing.run(...)
features = modules.cell_features.run(core_data=core, ...)
```

**Option 2: Pass checkpoint directory (auto-loads)**
```python
features = modules.cell_features.run(
    core_data="sprm_outputs",  # Loads from checkpoints/
    output_dir="sprm_outputs"
)
```

**Option 3: Explicit checkpoint loading**
```python
core = modules.preprocessing.load_checkpoint("sprm_outputs")
features = modules.cell_features.load_checkpoint("sprm_outputs")
```

### Force Re-run

To force recomputation, delete the checkpoint:
```python
import shutil
shutil.rmtree("sprm_outputs/checkpoints/cell_features.h5")
```

---

## Common Workflows

### Full Pipeline
```python
from sprm import modules

core = modules.preprocessing.run(img_file=..., mask_file=..., output_dir=...)
shape = modules.shape_analysis.run(core_data=core, output_dir=...)
spatial = modules.spatial_graphs.run(core_data=core, output_dir=...)
image_analysis = modules.image_analysis.run(core_data=core, output_dir=...)
features = modules.cell_features.run(core_data=core, shape_data=shape, output_dir=...)
clusters = modules.clustering.run(core_data=core, cell_features=features,
                                   shape_data=shape, output_dir=...)
```

### Minimal Pipeline (Fast)
```python
core = modules.preprocessing.run(img_file=..., mask_file=..., output_dir=...)
features = modules.cell_features.run(core_data=core, output_dir=...,
                                      compute_texture=False)
clusters = modules.clustering.run(core_data=core, cell_features=features,
                                   output_dir=...)
```

### Re-run Clustering Only
```python
# All previous modules already ran, checkpoints exist
clusters_v2 = modules.clustering.run(
    core_data="sprm_outputs",  # Auto-loads from checkpoints
    cell_features="sprm_outputs",
    shape_data="sprm_outputs",
    output_dir="sprm_outputs_v2",
    n_clusters_range=(5, 15)  # Different parameters
)
```

### Segmentation Evaluation Only
```python
core = modules.preprocessing.run(img_file=..., mask_file=..., output_dir=...)
seg_metrics = modules.segmentation_eval.run(core_data=core, output_dir=...,
                                             standalone=True)
```

---

## Migration from Legacy API

### Old way
```python
from sprm import SPRM

SPRM.main(
    img_dir="images/",
    mask_dir="masks/",
    output_dir="outputs/",
    processes=4
)
```

### New way (equivalent)
```python
from sprm import modules
from pathlib import Path

for img_file, mask_file in zip(image_files, mask_files):
    core = modules.preprocessing.run(img_file, mask_file, "outputs")
    shape = modules.shape_analysis.run(core, "outputs")
    spatial = modules.spatial_graphs.run(core, "outputs")
    image_analysis = modules.image_analysis.run(core, "outputs")
    features = modules.cell_features.run(core, "outputs", shape_data=shape)
    clusters = modules.clustering.run(core, features, shape, "outputs")
```

**Note**: Legacy API (`SPRM.py`) still works for backward compatibility.

---

## Best Practices

1. **Start with minimal pipeline** for quick results, add optional modules as needed
2. **Skip texture computation** (`compute_texture=False`) to save ~50% time if not needed
3. **Use checkpoints** when experimenting with clustering parameters
4. **Check module dependencies** - some modules require others
5. **Clean up checkpoints** periodically to save disk space
6. **Use debug=False** in production for cleaner output

---

## Troubleshooting

### "Checkpoint not found" error
- Ensure preprocessing ran successfully first
- Check that `output_dir/checkpoints/` exists

### Memory issues
- Run modules sequentially instead of keeping all data in memory
- Skip optional modules you don't need
- Use minimal pipeline instead of full pipeline

### Module won't run
- Check dependencies - does it need another module's output?
- Verify checkpoint files exist in `output_dir/checkpoints/`
- Try explicit loading: `module.load_checkpoint(dir)`

---

## API Reference Summary

### Data Classes
- `CoreData`: Preprocessing output
- `SegmentationMetrics`: Segmentation quality metrics
- `ShapeData`: Cell shape features
- `SpatialData`: Spatial relationships
- `ImageAnalysisData`: Image-level analyses
- `CellFeatures`: Per-cell features
- `ClusteringResults`: Clustering outputs

### Functions (all modules)
- `run(...)`: Execute module with given parameters
- `load_checkpoint(checkpoint_dir)`: Load saved checkpoint

### Utilities
- `CheckpointManager`: Low-level checkpoint operations

