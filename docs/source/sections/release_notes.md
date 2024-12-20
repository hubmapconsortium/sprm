# SPRM v1.5.1
December 20, 2024 Changes since v1.4.4

## New Features

[SPRM DEMO](https://github.com/murphygroup/SPRM/tree/master/demo)

We now feature a interactive demo that users can use to run SPRM and display it's functionalities. 

## Improvements

SPRM now takes in multiple cell type labels from different vendors/providers(e.g. multiple columns for each that identifies cell types) and will find corresponding subtypes for each representation. 

Example representation:
| Cell ID | Cell-Types 1| Cell-Types 2 | [Cell-Types 1]-tSNE_allfeatures-subtypes | [Cell-Types 1]-UMAP_allfeatures-subtypes | [Cell-Types 1]-mean-cell-subtypes | [Cell-Types 2]-tSNE_allfeatures-subtypes | [Cell-Types 2]-UMAP_allfeatures-subtypes | [Cell-Types 2]-mean-cell-subtypes |
| 69 | B Cells | T Cells | B:4 | B:3 | B:1 | T:1 | T:2 | T:3 |
| 71 | other cells | B Cells | o:1 | o:2 |o:2 | B:1 | B:2 | B:3 |
| 72 | other cells | B Cells | o:1 | o:2 |o:2 | B:4 | B:3 | B:2 |
| 74 | CD4-positive T Cells | B Cells | CD:1 | CD:2 | CD:2 | B:4 | B:3 | B:2 |

## Bug Fixes

Fix bug issue in reading 2D/3D images where channels were transformed incorrectly. 
Fix bug issue in OutlinePCA in which no rotation of cells are needed and therefore would skip rotation step.
Add edge case detection for cells that are under a specific pixel area - which is now defined in the options.


# SPRM v1.4.4
July 10, 2024 Changes since v1.2.1

## New Features/Outputs:
Identification of cell types and subtypes: A number of groups are working on ways of automating assignment of cell type labels to individual cells, especially in CODEX images.  These methods often use only a small number of markers available for a given dataset.  In order to be able to provide for future integration of the results of such methods into spatial proteomics pipelines, and to enhance them by identifying cell subtypes, new capabilities have been added.  These consist of:
- An optional input file containing cell type assignments in addition to the normal intensity image and mask image inputs is now accepted.
- A comparison of cell type labels to SPRM clustering results is done.  
        - For each feature set computed by SPRM (e.g., mean intensity of each channel per cell, total intensity of each channel per cell, covariance matrix of channel intensities per cell, cell shape features, tSNE and UMAP embeddings of all features), SPRM previously produced cell clusters using k-means clustering for various k values (range specified by the option “num_clusters) and silhouette scores to decide upon k.  [The results for different k values are saved if the debug option is true..]
        - SPRM now compares the clusters for each k with the cell type labels.  An overlap score is kept, and the results from the k value giving the best overlap (closest agreement) are kept.  These results are written to the “X-cluster_scores.csv” file, where “X” is the input filename..
        - Cell subtypes are then assigned by finding clusters that significantly overlap with each cell type. For illustration, cells that are assigned the label “B cells” might be found primarily in two clusters (e.g., cluster 3 and 5) using the mean channel intensities.  These would be assigned labels “B:3” and “B:5”.  
        - Additional columns are written to the “X-cell_cluster.csv” file containing assignments for each cell using the provided cell types and also the subtypes found for each feature subset. An example fragment from the “X-cell_cluster.csv” is shown below
 
| Cell ID | Cell-Types | tSNE_allfeatures-subtypes | UMAP_allfeatures-subtypes | mean-cell-subtypes |
| 69 | B Cells | B:4 | B:3 | B:1 |
| 71 | other cells | o:1 | o:2 |o:2 |
| 72 | other cells | o:1 | o:2 |o:2 |
| 74 | CD4-positive T Cells | CD:1 | CD:2 | CD:2 |


## SPRM 3D support: 

SPRM has always supported 3D images but this functionality has not been previously used in HuBMAP pipelines.  In preparation for use with 3D CODEX/Phenocycler datasets, proper operation for 3D images has been tested and minor issues fixed.  Some SPRM outputs are not yet supported for 3D images, including cell shape features and clustering and cell adjacency matrices and graphs.  For 3D images, the images of  cell cluster assignments that are generated as PNG (Portable Network Graphics) files for 2D images are generated as APNGs (Animated Portable Network Graphics) that show all Z slices.

## Improvements:
Colors of clusters in generated .png files are now consistent across different feature sets to the extent possible.
Performance for cell clustering has been improved by removing redundant function calls in outlinePCA.
Add option “image_analysis” to control whether to output images from pixel-wise analysis (Non-negative factorization, principal components analysis, pixel clustering).
Implement default options file to better organize options. 

## Bug Fixes:
Fix edge case of cell indexing
Fix truncation of S/N normalized intensities to integers when calculating features.
Fix redundant normalization of background intensity in reporting of ‘Cell / Background’ in Quality Metrics JSON.
Fix logic issue with APNG in which it 
