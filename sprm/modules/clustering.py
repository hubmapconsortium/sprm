"""
Module 6: Clustering & Cell Typing

Performs various clustering methods and cell type assignment.
This is the final analysis module.
"""

from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from ..SPRM_pkg import cell_analysis, read_options
from .checkpoint_manager import (
    CellFeatures,
    CheckpointManager,
    ClusteringResults,
    CoreData,
    ImageAnalysisData,
    ShapeData,
    SpatialData,
)

DEFAULT_OPTIONS_FILE = Path(__file__).resolve().parents[1] / "options.txt"


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
    num_markers: int = 3,
    *,
    debug: bool = False,
    skip_texture: Union[bool, Literal["auto"]] = "auto",
    run_outlinePCA: Union[bool, Literal["auto"]] = "auto",
    channel_label_combo: Union[List[str], Literal["auto"]] = "auto",
    tSNE_all_preprocess: List[str] = ["none", "zscore", "blockwise_zscore"],
    tSNE_all_ee: str = "default",
    tSNE_all_perplexity: int = 35,
    tSNE_all_tSNEInitialization: str = "pca",
    tsne_all_svdsolver4pca: str = "full",
    tSNE_texture_calculation_skip: int = 1,
    tSNE_num_components: int = 2,
    options: Optional[Union[Path, str, Dict[str, Any]]] = None,
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
    num_markers : int, default 3
        Number of marker channels to select per cluster (legacy `options["num_markers"]`).
    debug : bool, default False
        Enable debug output in legacy functions.
    skip_texture : bool or "auto", default "auto"
        Whether to skip texture-based clustering. If "auto", inferred from `cell_features.texture_vectors`.
    run_outlinePCA : bool or "auto", default "auto"
        Whether to run outline PCA clustering. If "auto", inferred from whether `shape_data` is provided.
    channel_label_combo : list[str] or "auto", default "auto"
        Flattened covariance feature names (cartesian product of channel labels).
        If "auto", computed from the image's channel labels.
    tSNE_all_preprocess : list[str], default ["none", "zscore", "blockwise_zscore"]
    tSNE_all_ee : str, default "default"
    tSNE_all_perplexity : int, default 35
    tSNE_all_tSNEInitialization : str, default "pca"
    tsne_all_svdsolver4pca : str, default "full"
    tSNE_texture_calculation_skip : int, default 1
    tSNE_num_components : int, default 2
        t-SNE configuration options passed through to legacy clustering code.
    options : Path, str, dict, or None, default None
        Optional legacy options source. Can be:
        - Path/str to an options.txt file (overlays package defaults)
        - Dict of legacy options (overlays package defaults)
        - None (use package defaults)

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

    # Create options dict for compatibility with legacy `SPRM_pkg.cell_analysis()`.
    # In the original pipeline, covariance legends rely on `options["channel_label_combo"]`
    # being set to the flattened covariance feature names (cartesian product of channels).
    channel_labels = im.get_channel_labels()
    computed_channel_label_combo = [
        ":".join(pair) for pair in product(channel_labels, channel_labels)
    ]

    # Build legacy options dict for compatibility with SPRM_pkg.cell_analysis().
    # Start from package defaults, then overlay user-provided options, then overlay explicit parameters.
    if options is None:
        compat_options: Dict[str, Any] = dict(read_options(DEFAULT_OPTIONS_FILE))
    elif isinstance(options, (str, Path)):
        compat_options = dict(read_options(Path(options), DEFAULT_OPTIONS_FILE))
    elif isinstance(options, dict):
        compat_options = dict(read_options(DEFAULT_OPTIONS_FILE))
        compat_options.update(options)
    else:
        raise TypeError(f"options must be None, Path, str, or dict, got {type(options)}")

    compat_options["num_cellclusters"] = [
        clustering_method,
        n_clusters_range[0],
        n_clusters_range[1],
        n_clusters_step,
    ]
    compat_options["num_shapeclusters"] = [
        clustering_method,
        n_shape_clusters_range[0],
        n_shape_clusters_range[1],
        n_clusters_step,
    ]

    # Used by SPRM_pkg.findmarkers() to pick the top marker channels per cluster.
    compat_options["num_markers"] = num_markers

    # Defaults: match the behavior this module previously hard-coded.
    compat_options["debug"] = debug

    computed_skip_texture = cell_features.texture_vectors is None or (
        cell_features.texture_vectors.sum() == 0
    )
    compat_options["skip_texture"] = (
        computed_skip_texture if skip_texture == "auto" else skip_texture
    )

    computed_run_outlinePCA = shape_data is not None
    compat_options["run_outlinePCA"] = (
        computed_run_outlinePCA if run_outlinePCA == "auto" else run_outlinePCA
    )

    compat_options["channel_label_combo"] = (
        computed_channel_label_combo if channel_label_combo == "auto" else channel_label_combo
    )

    # Defaults are now expressed in the function signature.
    compat_options["tSNE_all_preprocess"] = tSNE_all_preprocess
    compat_options["tSNE_all_ee"] = tSNE_all_ee
    compat_options["tSNE_all_perplexity"] = tSNE_all_perplexity
    compat_options["tSNE_all_tSNEInitialization"] = tSNE_all_tSNEInitialization
    compat_options["tsne_all_svdsolver4pca"] = tsne_all_svdsolver4pca
    compat_options["tSNE_texture_calculation_skip"] = tSNE_texture_calculation_skip
    compat_options["tSNE_num_components"] = tSNE_num_components

    print(f"Clustering configuration:")
    print(f"  Cluster range: {n_clusters_range[0]}-{n_clusters_range[1]}")
    print(f"  Method: {clustering_method}")
    print(f"  Shape clustering: {'enabled' if shape_data else 'disabled'}")
    print(
        f"  Texture features: {'enabled' if not compat_options['skip_texture'] else 'disabled'}"
    )
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
        options=compat_options,
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

