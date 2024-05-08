# SPRM Preprocessing & Analysis

The following are the analysis steps that is within SPRM for a provided dataset.

## Preprocessing

SPRM performs several preprocessing steps to be able to efficiently process large dataset and normalize the images to allow for
comparisons of outputs across processed datasets. 

### Image Preprocessing

#### Background Normalization

The multiplex expression image is normalized by the background intensity and therefore converted to a signal to noise ratio.

### Segmentation Preprocessing

Segmentation preprocessing are conducted on the given input segmentation mask

#### Segmentation Regions of Interest

One of the main boost in efficiency in SPRM is that the segmentation mask is transformed in a pixel coordinate matrix in which for each cell the pixel coordinates are stored. This allows for fast and efficient look up of specific cells to enhance downstream analysis.

#### Edge Cell Detection

Cells that are on the edge or touching the boundary of the image are removed from considerations of the downstream analysis. This is due to the fact that those cells might not be a full representation and comparable to their other constituents witin the image.

## Image Analysis 

There are three main image analysis conducted within SPRM on expression 
image that has been normalized by background.

#### Top 3 Channel PCA (Principle Component)

Converting multiple channels into the top 3 principal components of channel intensity (finding major components of variation in the multiple channels).

#### Superpixel 

Converting pixels into “superpixels” by clustering them based on intensities in all channels.

#### NMF (Non-negative Matrix Factorization)

**V = W x H** 

V represents the pixel base image which is factorized into W x H matrices with the property that all three matrices do not have negative elements. This is used for dimensionality reduction on the image useful for parts based representation. 

## Mask Analysis

### Cell Adjacency

Finds neighborhood cells that are within a threshold distance of each other. The threshold distance is a user defined input in the options.
There is a matrix and graphical image that is produced from this step.

## Downstream Analysis (Statistical, Spatial, Clustering, and Subtyping)

### Features

Here are the main features that we calculate to use for our analysis per segmentation region (cell, cell boundaries, nucleus, and nuclear boundaries):

- Mean Intensity 
- Total Intensity
- Covariance of Intensity between Channels

### Spatial & Shape

#### Cell Adjacency
Users can input a dilation and delta parameters that adjusts for the threshold of what cells are considered to be in a neighborhood and adjacent to each other. The result is an adjacency graph which is based on the cell center and connects other cells based on the predefined parameters. 
In addition there is a adjacency matrix that is built that gives the distance between the adjacent cells in pixel units.

#### Shape Outlines
SPRM retrieves the outline of cells and conducts shape analysis to normalize the size of a cell and uses PCA to reduce it to a vector that encodes the shape variations.

### Clustering

SPRM will use KMeans unsupervised clustering to cluster based off of the main statistical features: mean, total, covariance, along with shape of the cells, UMAP and tSNE. 
UMAP and tSNE are conducted from the full feature sets which we refer to as the statistical features of the segmentation regions + the shape. 

### Subtyping

If provided an associated subtype identification that is able to be mapped to the corresponding segmentation mask, SPRM is able to conduct subtype analysis. 

The subtype analysis includes finding sub groups of cell types that exists within the image that is based on previous cluster based analysis on the full feature sets.

