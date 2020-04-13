# SPRM - Spatial Process & Relationship Modeling
Ted Zhang and Bob Murphy, Carnegie Mellon University
V0.51 April 7,2020

## Description
SPRM is a statistical modeling program used to calculate a range of descriptors/features from multichannel cell images.  It uses these descriptors for clustering of cells and saves them for use for comparison and content-based search of the images.  It can be used with any type of multichannel 2D or 3D image (e.g., CODEX, IMS).

## Inputs

Two OMETIFF files from the output of CytoKit -
4D Multiplexed intensity Image (3D for multiple channels)
4D Indexed Image (3D for multiple segmentations) containing one channel for each type of segmentation (currently “cells”, “nucleus”, “cell membrane” and “nuclear membrane”).

## Execution: (assuming SPRM.py is in working directory)
```bash
[python_path] SPRM.py [img_dir_path] [mask_dir_path] [options_path]
```

SPRM takes in three command line arguments that specify the path in the following order:
Image directory path
Mask directory path
Options file path

## Analyses: Descriptors

(* items are in progress)
Converting multiple channels into the top 3 principal components of channel intensity (finding major components of variation in the multiple channels) (output: OME-TIFF, “xxx-channel_pca.ome.tiff”)
Converting pixels into “superpixels” by clustering them based on intensities in all channels (output: OME-TIFF, “xxx-superpixel.ome.tiff”)
Calculating mean intensities in each channel for each segmentation (output: CSV (one for each segmentation),  “xxx_channel_mean.csv”, where xxx is the segmentation name; each CSV contains number of objects (e.g., cells)) vs number of channels **comparable between images**
Calculating total intensities in each channel for each segmentation (output: CSV (one for each segmentation), “xxx_channel_total.csv”, where xxx is the segmentation name; each CSV contains number of objects (e.g., cell)) vs number of channels **comparable between images**
Calculating covariance of intensities for all channels for each segmentation (output: CSV (one for each segmentation), “xxx_channel_covar.csv”, where xxx is the segmentation name; each CSV contains number of objects (e.g., cells) by number of channels squared (flattened) **comparable between images**
Calculating cell shape descriptors (using PCA of parametric outlines) (output: CSV, “cell_shape.csv”; number of cells by number of shape descriptors) **comparable between images**
*Calculating descriptors of subcellular location patterns in each cell (using texture and object features) (output: CSV, “cell_slf.csv” number of cells by number of features, one block per channel) **comparable between images**
*Calculating distance matrix for each cell (using subcellular location features) (output: CSV, “dist_cell_patternslf.csv” blocks of number of cells by number of cells, one block per channel)
*Calculating distance matrix for patterns in each cell (using transport-based morphometry) (output: CSV, ”dist_cell_patterntbm.csv” blocks of number of cells by number of cells, one block per channel)
*Producing channel spatial dependency graph for whole image (output: DOT, “sdg_image.dot”) **comparable between images**
*Producing channel spatial dependency graph for each cell (output: DOT, “sdg_cell.dot” one digraph for each cell) **comparable between images**

Note that the descriptors/features above flagged with **comparable between images** are “global” and can be used for comparison or searching between images (e.g., find cells in other images that are similar to this cell).  Others are only for comparison of different cells within an image (“local”).

## Analyses: Clustering

Clustering cells by a) mean intensity, b) total intensity, c) covariance, and d) concatenation of intensities and covariances
Clustering cells by cell shape descriptors
Clustering cells by subcellular pattern using features
Clustering cells after embedding distance matrix
Clustering by all descriptors

A single “master” clustering CSV file (“cell_cluster.csv”) is output containing the cluster number of each cell for all clusterings (one row for each cell, one column for each clustering method). For each clustering, the major independent contributors (“markers”) to the clusters (which must be measured features, not derived variables like principal components) are found.  

One “legend” CSV is produced for each clustering method that contains the mean values of each marker for each cluster (number of cells by number of markers).  This is intended to accompany display of the clustering results by showing the major contributors to the choice of the clusters.  One PNG is also produced for each clustering that shows the image with each cell colored by cluster number (this is mainly provided for checking downstream display of clusters).

## Outputs:

The outputs can be grouped into six types
OME-TIFFs showing pixel level results (remapping of channels) [2 per input image]
CSVs containing features for each cell (in some cases for each nucleus, cell membrane and nuclear membrane also) [16 per input image]
Master CSV containing clustering results for each cell (row) for each method (column) [1 per input image]
CSVs containing mean values of “markers” for each cluster for each clustering method [8 per input image]
PNGs showing coloring by cluster [8 per input image]
DOTs showing channel spatial dependency graphs [2 per input image]

## Prerequisites
Aicsimageio
Numpy
Sklearn
Pandas
Matplotlib

## Contact
Robert F. Murphy - murphy@cmu.edu
Ted (Ce) Zhang - tedz@andrew.cmu.edu
