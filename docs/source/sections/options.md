# SPRM Options Parameters

Generally the options are preset for every run and should not be changed. These are located as a separate file that SPRM will read. **These should not be confused with command line arguments**.

## SPRM Options Table

| Option | Default Value | Description |
|--------|---------------|-------------|
| **General/runtime controls** |
| `debug` | 0 | Enable verbose/diagnostic mode. Also disables subprocess isolation for multi-image runs. |
| `image_analysis` | 1 | Turn on the main image feature extraction/analysis steps. Set 0 to skip heavy analysis. |
| **Input, dimensionality, and selection** |
| `image_dimension` | 2D | Processing mode: 2D or 3D. Affects adjacency construction and slice handling. |
| `zslices` | 0 | Number of z-slices to consider in 3D. 0 lets the pipeline pick a best 2D slice when applicable. |
| `skip_empty_mask` | 1 | Skip images with no valid cells in the mask. |
| `interior_cells_only` | 1 | Use only interior cells for downstream feature computation (e.g., texture). |
| `valid_cell_threshold` | 10 | Minimum pixel count for a cell to be considered valid. |
| **Normalization and preprocessing** |
| `normalize_bg` | 1 | Apply background normalization across channels before feature extraction. |
| `zscore_norm` | 1 | Apply z-score normalization to features prior to clustering/embedding. |
| `tSNE_all_preprocess` | none zscore blockwise_zscore | Preprocessing for tSNE "all features": choose one of none, zscore, blockwise_zscore. blockwise_zscore balances blocks (cov/total/meanAll/shape/texture) before concatenation. |
| `tSNE_texture_calculation_skip` | 1 | Exclude texture features from the concatenated feature matrix used for tSNE. |
| **Texture (GLCM) features** |
| `skip_texture` | 1 | Skip GLCM texture computation entirely. |
| `glcm_angles` | [0] | Angular offsets (degrees) for GLCM. Currently only 0° is supported. |
| `glcm_distances` | [1] | Pixel distances for GLCM co-occurrence. Multiple distances expand features per channel/mask. |
| **Shape analysis and outlines** |
| `run_outlinePCA` | 1 | Compute PCA on resampled cell outlines to derive shape features. |
| `num_outlinepoints` | 100 | Number of points used to resample cell outlines prior to PCA. |
| `num_shapeclusters` | silhouette 3 6 1 | Shape clustering configuration: method (silhouette) with min=3, max=6, keep=1. The best K by silhouette is selected in-range. |
| **Clustering (cells/voxels/markers)** |
| `num_cellclusters` | silhouette 3 10 1 | Cell-feature clustering: method (silhouette) with min=3, max=10, keep=1. Best K chosen by silhouette. |
| `num_voxelclusters` | 3 | Number of clusters for voxel-level grouping (e.g., intensity/region partitioning). |
| `num_markers` | 3 | Number of markers to include for marker-driven analyses/plots. |
| `recluster` | 0 | If enabled, perform a secondary clustering pass after feature updates/refinements. |
| **Dimensionality reduction (tSNE and PCA before tSNE)** |
| `tSNE_num_components` | 2 | Number of components for tSNE output (and PCA components when PCA init is used). |
| `tSNE_all_tSNEInitialization` | pca | Initialization mode for tSNE "all features". pca triggers a PCA transform before tSNE. |
| `tsne_all_svdsolver4pca` | full | PCA SVD solver when using PCA init: full or randomized (auto-retries if full fails). |
| `tSNE_all_perplexity` | 35 | tSNE perplexity hyperparameter. |
| `tSNE_all_ee` | default | tSNE early exaggeration. default sets it to roughly N/10; numeric values are accepted. |
| `num_channelPCA_components` | 3 | Number of PCA components per channel or stage where per-channel PCA is applied (used in channel-wise reductions feeding tSNE). |
| **Cell adjacency graph and neighborhood** |
| `cell_graph` | 1 | Build the cell adjacency graph and related sparse distance matrices. |
| `cell_adj_parallel` | 0 | Use parallel/numba-optimized windowing for adjacency (1) or standard Python (0). |
| `cell_adj_dilation_itr` | 3 | Number of binary dilation iterations used to dilate cell edges when checking contact/neighbors. |
| `adj_matrix_delta` | 3 | Window padding around each cell's ROI to restrict adjacency calculations and speed up searches. |
| **Reallocation/refinement controls** |
| `reallocation_descent_rate` | 0.1 | Step size for iterative membership/label reallocation during post-cluster refinement. |
| `reallocation_quit_criterion` | 0.0001 | Convergence tolerance for reallocation iterations. |
| `num_of_reallocations` | 1 | Maximum number of reallocation passes. |
| `reallocation_avg_bg` | 1 | Include/weight background averaging during reallocation updates. |
| **Evaluation, visualization, and misc** |
| `sprm_segeval_both` | 2 | Segmentation evaluation mode; compute metrics for multiple mask channels/variants. |
| `apng_delay` | 100 | Delay (ms) between frames for generated animated PNGs. |
| `subtype_thresh` | 0.1 | Threshold for assigning cell subtypes from provided labels/scores. |

---

## Additional Information

The cell segmentation masks may be produced by any of a number of methods (see Chen & Murphy (1) for evaluation of a number of methods), but there must be a corresponding nuclear mask for each cell mask whose indices must be the same. In case a particular segmentation method does not always produce matched masks, a precursor "mask repair" program can be run before running SPRM (1). This precursor step modifies the mask files by removing any objects in the cell or nuclear masks that do not match and selecting the largest nuclear object if more than one is present for a given cell. It will also trim the nuclear masks to remove any pixels on or outside of the cell membrane of their corresponding cell mask. If masks for the cell boundary and the nuclear boundary are not present, they will be generated.

Depending on whether the input images are three-dimensional, options may be provided to specify a subset of z slices for a given image dataset. This is especially relevant for (CODEX) datasets since while these are acquired as a stack of 2D images many of these are out of focus.

### SPRM Methods and Example Outputs

#### Segmentation Quality Metrics
A critical measure of image quality is how well the image can be segmented into single cells. SPRM therefore reports segmentation quality metrics and an overall segmentation quality score using the CellSegmentationEvaluator package (https://github.com/murphygroup/CellSegmentationEvaluator) (1). The metrics are based on a series of assumptions about desired characteristics of a good segmentation, such as consistency of marker expression with a cell type.

#### Preprocessing
After the segmentation quality metrics are calculated, SPRM performs preprocessing to prepare for subsequent analysis. Any cells in the mask that are touching an edge of the image are removed, since accurate quantification cannot be achieved for these partial cells. Then, for each channel a background intensity is found (as the average of all pixels outside cells) and channel intensities are normalized to signal to noise ratios by dividing each channel by its respective background intensity.