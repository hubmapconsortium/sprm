# SPRM Options Parameters

Generally the options are preset for everyrun and should not be changed. These are located as a seperate file that SPRM will read. **These should not be confused with commandline arguments**.

Default Options
```
debug 0
image_analysis 1
num_cellclusters silhouette 3 10 1
num_voxelclusters 3
num_markers 3
num_channelPCA_components 3
num_outlinepoints 100
num_shapeclusters silhouette 3 6 1
precluster_sampling 0.07
precluster_threshold 10000
skip_empty_mask 1
reallocation_descent_rate 0.1
reallocation_quit_criterion 0.0001
num_of_reallocations 1
reallocation_avg_bg 1
zslices 0
recluster 0
glcm_angles [0]
glcm_distances [1]
run_outlinePCA 1
interior_cells_only 1
normalize_bg 1
zscore_norm 1
skip_texture 1
valid_cell_threshold 10
tSNE_all_preprocess none zscore blockwise_zscore
tSNE_all_ee default
tSNE_all_perplexity 35
tSNE_all_tSNEInitialization pca
tsne_all_svdsolver4pca full
cell_adj_parallel 0
cell_graph 1
cell_adj_dilation_itr 3
sprm_segeval_both 2
tSNE_texture_calculation_skip 1
tSNE_num_components 2
adj_matrix_delta 3
image_dimension 2D
apng_delay 100
subtype_thresh 0.1
```

---

## Options

Main options and purpose is listed here. Not all options are listed as some are not currently being used in the main pipeline but will have further usage in the future.

### debug
Purpose: Enables debug to provide more information as SPRM runs.
Type: Int 
Default: 0

### image_dimension
Purpose: Tells SPRM if this is a 2D or 3D image.
Type: Str
Default: 2D

### image_analysis
Purpose: Conduct just image analysis which results in Top 3 Channel PCA, Superpixel, and Non-negative matric factorization results.
Type: Int
Default: 1

### run_outlinePCA
Purpose: run outlinePCA which is the module responsible for producing cell outlines.
Type: Int
Default: 1

### interior_cells_only
Purpose: Only process interior cells, cells on the edge of the image will not be processed.
Type: Int
Default: 1

### normalize_bg
Purpose: Normalize the background intensity but dividing the total intensity of the image (per channel) by the average background intensity
Type: Int
Default: 1

### zscore_norm
Purpose: Normalize the channel intensities by z-score method.
Type: Int
Default: 1

### valid__cell_threshold
Purpose: Defines the threshold of size in pixel for a valid cell to be included in processing
Type: Int
Default: 10

### cell_graph
Purpose: Runs the cell graph module to produce adjacency matrix of which cells neighbor each other as well as an associated image.
Type: Int
Default: 1