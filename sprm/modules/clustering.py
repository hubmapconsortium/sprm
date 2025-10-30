"""
Module 6: Clustering & Cell Typing

Performs various clustering methods and cell type assignment.
This is the final analysis module.
"""

from pathlib import Path
from typing import Optional, Union

from ..SPRM_pkg import cell_analysis
from .checkpoint_manager import (
    CellFeatures,
    CheckpointManager,
    ClusteringResults,
    CoreData,
    ImageAnalysisData,
    ShapeData,
    SpatialData,
)


def run(
    core_data: Union[Path, str, CoreData],
    cell_features: Union[Path, str, CellFeatures],
    output_dir: Union[Path, str],
    shape_data: Optional[Union[Path, str, ShapeData]] = None,
    spatial_data: Optional[Union[Path, str, SpatialData]] = None,
    image_analysis_data: Optional[Union[Path, str, ImageAnalysisData]] = None,
    celltype_labels: Optional[Path] = None,
    n_clusters_range: tuple = (3, 10),
    n_clusters_step: int = 1,
    clustering_method: str = "silhouette",
    n_shape_clusters_range: tuple = (3, 6),
) -> ClusteringResults:
    """
    Perform clustering analysis on cell features.

    Multiple clustering methods are applied:
    - K-means on mean intensities
    - K-means on covariance matrices
    - K-means on total intensities
    - K-means on texture features
    - K-means on shape features (if available)
    - t-SNE + clustering on all features
    - UMAP + clustering on all features

    Parameters:
    -----------
    core_data : Path, str, or CoreData
        Either a CoreData object or path to checkpoint directory
    cell_features : Path, str, or CellFeatures
        Either a CellFeatures object or path to checkpoint directory
    output_dir : Path or str
        Directory for outputs
    shape_data : Path, str, ShapeData, or None
        Optional shape features for shape-based clustering
    spatial_data : Path, str, SpatialData, or None
        Optional spatial data (not directly used in clustering but saved in outputs)
    image_analysis_data : Path, str, ImageAnalysisData, or None
        Optional image-level data (not directly used in clustering)
    celltype_labels : Path or None
        Optional CSV file with cell type labels for supervised assignment
    n_clusters_range : tuple, default (3, 10)
        Range of cluster numbers to try (min, max)
    n_clusters_step : int, default 1
        Step size for cluster number search
    clustering_method : str, default "silhouette"
        Method to select optimal cluster number: "silhouette" or "fixed"
    n_shape_clusters_range : tuple, default (3, 6)
        Range of cluster numbers for shape clustering

    Returns:
    --------
    ClusteringResults: Cluster assignments and centers

    Outputs:
    --------
    - {image_name}-{channel}_clusterIDs.csv
    - {image_name}-Legend_{method}.csv (for each clustering method)
    - {image_name}-{method}_clusters.png (visualization for each method)
    - {image_name}-all_features.json
    - {image_name}-clustering_scores.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-load all data
    core_data = CheckpointManager.auto_load_core_data(core_data, output_dir)
    cell_features = CheckpointManager.auto_load_cell_features(cell_features, output_dir)
    shape_data = CheckpointManager.auto_load_shape_data(shape_data, output_dir)
    spatial_data = CheckpointManager.auto_load_spatial_data(spatial_data, output_dir)
    image_analysis_data = CheckpointManager.auto_load_image_analysis_data(
        image_analysis_data, output_dir
    )

    print("=" * 60)
    print("SPRM Module 6: Clustering & Cell Typing")
    print("=" * 60)

    im = core_data.im
    mask = core_data.mask
    bestz = core_data.bestz
    cellidx = core_data.cell_index
    baseoutputfilename = im.get_name()

    # Get cell segmentation label
    seg_n = mask.get_labels("cell")

    # Load cell type labels if provided
    celltype_labels_df = None
    if celltype_labels:
        import pandas as pd

        celltype_labels_path = Path(celltype_labels)
        print(f"Loading cell type labels from: {celltype_labels_path.name}")
        celltype_labels_df = pd.read_csv(celltype_labels_path)

    # Create options dict for compatibility
    options = {
        "num_cellclusters": [
            clustering_method,
            n_clusters_range[0],
            n_clusters_range[1],
            n_clusters_step,
        ],
        "num_shapeclusters": [
            clustering_method,
            n_shape_clusters_range[0],
            n_shape_clusters_range[1],
            n_clusters_step,
        ],
        "debug": False,
        "skip_texture": cell_features.texture_vectors is None
        or cell_features.texture_vectors.sum() == 0,
        "run_outlinePCA": shape_data is not None,
        "channel_label_combo": [],  # Will be populated by cell_analysis
        "tSNE_all_preprocess": ["none", "zscore", "blockwise_zscore"],
        "tSNE_all_ee": "default",
        "tSNE_all_perplexity": 35,
        "tSNE_all_tSNEInitialization": "pca",
        "tsne_all_svdsolver4pca": "full",
        "tSNE_texture_calculation_skip": 1,
        "tSNE_num_components": 2,
    }

    print(f"Clustering configuration:")
    print(f"  Cluster range: {n_clusters_range[0]}-{n_clusters_range[1]}")
    print(f"  Method: {clustering_method}")
    print(f"  Shape clustering: {'enabled' if shape_data else 'disabled'}")
    print(f"  Texture features: {'enabled' if not options['skip_texture'] else 'disabled'}")
    print(f"  Cell type labels: {'provided' if celltype_labels_df is not None else 'not provided'}")

    # Initialize list for storing cluster dataframes
    df_all_cluster_list = []

    # Extract shape vectors if available
    shape_vectors = shape_data.shape_vectors if shape_data else None
    norm_shape_vectors = shape_data.norm_shape_vectors if shape_data else None

    # Run comprehensive cell analysis with all clustering methods
    print("\nRunning clustering analysis...")
    print("This includes:")
    print("  - K-means on mean, covariance, total intensities")
    print("  - K-means on texture features")
    if shape_data:
        print("  - K-means on shape features")
    print("  - t-SNE dimensionality reduction + clustering")
    print("  - UMAP dimensionality reduction + clustering")
    print()

    cell_analysis(
        im=im,
        mask=mask,
        filename=baseoutputfilename,
        bestz=bestz,
        output_dir=output_dir,
        seg_n=seg_n,
        cellidx=cellidx,
        options=options,
        celltype_labels=celltype_labels_df,
        df_all_cluster_list=df_all_cluster_list,
        mean_vector=cell_features.mean_vector,
        covar_matrix=cell_features.covar_matrix,
        total_vector=cell_features.total_vector,
        texture_vectors=cell_features.texture_vectors,
        texture_channels=cell_features.texture_channels,
        shape_vectors=shape_vectors,
        norm_shape_vectors=norm_shape_vectors,
    )

    print("\n✓ Clustering analysis complete")
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 60)

    # Return clustering results (actual data is in the saved files)
    # The cell_analysis function saves everything to disk
    return ClusteringResults(
        cluster_assignments={},  # Saved to CSV files
        cluster_centers={},  # Saved to CSV files
        cluster_scores=None,  # Saved to CSV files
        celltype_assignments=celltype_labels_df,
    )

