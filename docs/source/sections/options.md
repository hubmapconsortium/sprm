# SPRM Options Parameters

Generally the options are preset for every run and should not be changed. These are located as a separate file that SPRM will read. **These should not be confused with command line arguments**.

## SPRM Options Table

| Option | Default Value | Description |
|--------|---------------|-------------|
| **General/runtime controls** |
| `debug` | 0 | Enable verbose/diagnostic mode. Also disables subprocess isolation for multi-image runs. |
| `image_analysis` | 1 | Perform the main image feature-extraction and analysis steps. |
| **Dimensionality selection and preprocessing** |
| `image_dimension` | 2D | Processing mode: 2D or 3D. The default for 3D input
images is to select one z-slice to process (see option
zslices). Therefore, image_dimension must be set to 3D
to process all slices. |
| `zslices` | 0 | For image_dimension=2D, specifies which z-slice to
process (setting to 0 picks the slice with the highest total
intensity). Ignored for image_dimension=3D. |
| `skip_empty_mask` | 1 | Skip images with no valid cells in the mask. |
| `interior_cells_only` | 1 | Use only interior cells for downstream feature
computation (i.e., remove cells touching image edge). |
| `valid_cell_threshold` | 10 | Minimum pixel count for a cell to be considered valid. |
| `normalize_bg` | 1 | Apply background normalization across channels before feature extraction. |
| **Cell shape analysis** | 
| `run_outlinePCA` | 1 | Find cell outlines and use PCA to derive shape features. |
| `num_outlinepoints` | 100 | Number of points used to resample cell outlines prior to PCA. |
| **Clustering (cells/voxels/markers)** |
| `zscore_norm` | 1 | Apply z-score normalization to features prior to clustering/embedding. |
| `tSNE_all_preprocess` | none | Define desired preprocessing for tSNE: “none” (don’t zscore), “zscore” (zscore all features together), or “blockwise_zscore” (zscore each block of features (cov/total/meanAll/shape) separately. |
| `num_shapeclusters` | silhouette 3 6 1 | Control shape clustering (same structure as num_cellclusters but different default max). |
| `num_cellclusters` | silhouette 3 10 1 | Control choice of number of cell clusters found by kmeans: must begin with method (only method currently supported is silhouette) followed by min and max k followed by keep (1=output clustering resuts for each k). |
| `num_voxelclusters` | 3 | Number of clusters for voxel-level grouping (e.g., intensity/region partitioning). |
| `num_markers` | 3 | Number of markers to include for marker-driven analyses/plots. |
| **Dimensionality reduction (tSNE and PCA before tSNE)** |
| `tSNE_num_components` | 2 | Number of components for tSNE output (and PCA components when PCA init is used). |
| `tSNE_all_tSNEInitialization` | pca | Perform PCA transform before tSNE. |
| `tsne_all_svdsolver4pca` | full | PCA SVD solver when using PCA init: full or randomized (auto-retries if full fails). |
| `tSNE_all_perplexity` | 35 | tSNE perplexity hyperparameter. |
| `tSNE_all_ee` | default | tSNE early exaggeration. default sets it to roughly N/10; numeric values are accepted. |
| `num_channelPCA_components` | 3 | When creating pixel/voxel-wise colored images, number of PCA or k-means components to choose. |
| **Cell adjacency graph and neighborhood** |
| `cell_graph` | 1 | Build the cell adjacency graph and related sparse distance matrices. |
| `cell_adj_parallel` | 0 | Use parallel/numba-optimized windowing for adjacency (1) or standard Python (0). |
| `cell_adj_dilation_itr` | 3 | Number of binary-dilation iterations done before checking neighbors. |
| `adj_matrix_delta` | 3 | Padding (in pixels) to add around each cell’s bounding box. For efficiency, only pairs of cells with overlapping
padded bounding boxes are checked for potential neighbors. |
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