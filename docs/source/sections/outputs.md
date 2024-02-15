# SPRM Outputs

The following are descriptions of SPRM outputs related to their reg expression naming (xxx).

The outputs can be grouped into x types

### OME-TIFFs showing pixel level results (remapping of channels) [3 per input image]

- xxx-channel_pca.ome.tiff
- xxx-superpixel.ome.tiff
- xxx-nmf_calculation.ome.tiff

### CSV showing interpolated cell outlines & polygons [2 per input image]
 - xxx-cell_shape.csv
 - xxx-cell_polygons_spatial.csv

### CSVs containing features for each cell [4 per image]
 - xxx_texture.csv 
 - xxx_cell_channel_meanALL.csv 
 - xxx-cell_adj_list.csv  
 - xxx-cell_centers.csv

### CSVs containing features for each segmentation [12 per image]
 - xxx-[segmentation channel]_channel_covar.csv 
 - xxx-[segmentation channel_channel_mean.csv 
 - xxx-[segmentation channel]l_channel_total.csv

### Master CSV containing clustering results for each cell (row) for each method (column) [1 per input image]
 - xxx-cell_cluster.csv
 - 
### CSVs containing mean values of “markers” for each cluster for each clustering method [5 per input image]
 - xxx-clustercells_cellcovariance_legend.csv 
 - xxx-clustercells_cellmeanALL_legend.csv  
 - xxx-clustercells_cellmean_legend.csv 
 - xxx-clustercells_cellshape_legend.csv 
 - xxx-clustercells_celltotal_legend.csv

### PNGs showing coloring by cluster [7 per input image]
 - xxx-ClusterByCovarPerCell.png
 - xxx-ClusterByMeansAllMasks.png 
 - xxx-ClusterByMeansPerCell.png
 - xxx-ClusterByShape.png 
 - xxx-ClusterByTotalPerCell.png 
 - xxx-Superpixels.png 
 - xxx-Top3ChannelPCA.png

### CSV containing the signal to noise ratios of the image per channel [1 per input image]
 - xxx-SNR.csv

### CSV containing PCA and Silhouette analysis of the image [2 per input image]
 - xxx-channelPCA_summary.csv 
 - xxxclusteringsilhouetteScores.csv

### CSV containing the summary file [3 per run]
 - summary_otsu.csv
 - summary_zscore.csv
 - SPRM_Image_Quality_Measures.json
