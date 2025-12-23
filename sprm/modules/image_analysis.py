"""
Module 4b: Image-Level Analysis

Performs whole-image analyses: NMF, superpixel clustering, channel PCA.
This module is optional.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..SPRM_pkg import (
    NMF_calc,
    clusterchannels,
    plot_img,
    plotprincomp,
    read_options,
    voxel_cluster,
    write_ometiff,
)
from .checkpoint_manager import CheckpointManager, CoreData, ImageAnalysisData


DEFAULT_OPTIONS_FILE = Path(__file__).resolve().parents[1] / "options.txt"


def run(
    core_data: Union[Path, str, CoreData],
    output_dir: Union[Path, str],
    n_voxel_clusters: int = 3,
    n_channel_pca_components: int = 3,
    *,
    precluster_sampling: Optional[float] = None,
    precluster_threshold: Optional[int] = None,
    zscore_norm: Optional[Union[bool, int]] = None,
    debug: Optional[bool] = None,
    image_dimension: Optional[str] = None,
    options: Optional[Union[Path, str, Dict[str, Any]]] = None,
    compute_nmf: bool = True,
    generate_visualization_tiff: bool = True,
) -> ImageAnalysisData:
    """
    Perform image-level analyses: NMF, superpixels, channel PCA.

    Parameters:
    -----------
    core_data : Path, str, or CoreData
        Either a CoreData object or path to checkpoint directory
    output_dir : Path or str
        Directory for outputs and checkpoints
    n_voxel_clusters : int, default 3
        Number of clusters for superpixel/voxel clustering
    n_channel_pca_components : int, default 3
        Number of PCA components to compute for channels
    precluster_sampling : float, optional
        Fraction of voxels sampled for the initial KMeans fit (used by `SPRM_pkg.voxel_cluster()`).
        If None, uses `options["precluster_sampling"]` or package defaults.
    precluster_threshold : int, optional
        Minimum number of voxels to sample for the initial KMeans fit (used by `SPRM_pkg.voxel_cluster()`).
        If None, uses `options["precluster_threshold"]` or package defaults.
    zscore_norm : bool or int, optional
        Whether to z-score normalize voxel intensities before clustering (used by `SPRM_pkg.voxel_cluster()`).
        If None, uses `options["zscore_norm"]` or package defaults.
    debug : bool, optional
        Enable debug output in legacy functions.
        If None, uses `options["debug"]` or package defaults.
    image_dimension : str, optional
        Image dimensionality (e.g. "2D" or "3D") passed to legacy plotting/writing utilities.
        If None, uses `options["image_dimension"]` or package defaults.
    options : Path, str, dict, or None, default None
        Optional legacy options source. Can be:
        - Path/str to an options.txt file (overlays package defaults)
        - Dict of legacy options (overlays package defaults)
        - None (use package defaults)
    compute_nmf : bool, default True
        Whether to compute non-negative matrix factorization
    generate_visualization_tiff : bool, default True
        Whether to generate visualization OME-TIFF files

    Returns:
    --------
    ImageAnalysisData: NMF results, superpixels, PCA components

    Checkpoints saved:
    ------------------
    - checkpoints/image_analysis.pkl

    Outputs:
    --------
    - {image_name}-Superpixels.png
    - {image_name}-Top3ChannelPCA.png
    - {image_name}-PCA_silhouette.csv
    - {image_name}-NMF_*.csv
    - {image_name}-visualization.ome.tiff (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-load core data if needed
    core_data = CheckpointManager.auto_load_core_data(core_data, output_dir)

    # Check if checkpoint already exists
    if CheckpointManager.exists_image_analysis_data(output_dir):
        print(f"Loading existing image analysis checkpoint from {output_dir}")
        return CheckpointManager.load_image_analysis_data(
            CheckpointManager.get_checkpoint_dir(output_dir)
        )

    print("=" * 60)
    print("SPRM Module 4b: Image-Level Analysis")
    print("=" * 60)

    im = core_data.im
    bestz = core_data.bestz
    interior_cells = core_data.interior_cells
    baseoutputfilename = im.get_name()

    # Build legacy options dict for compatibility with SPRM_pkg.* utilities.
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

    # Required knobs for this module
    compat_options["num_voxelclusters"] = n_voxel_clusters
    compat_options["num_channelPCA_components"] = n_channel_pca_components

    # Defaults: match the behavior this module previously hard-coded.
    compat_options["precluster_sampling"] = (
        0.07 if precluster_sampling is None else precluster_sampling
    )
    compat_options["precluster_threshold"] = (
        10000 if precluster_threshold is None else precluster_threshold
    )
    compat_options["zscore_norm"] = 1 if zscore_norm is None else zscore_norm
    compat_options["debug"] = False if debug is None else debug
    compat_options["image_dimension"] = "2D" if image_dimension is None else image_dimension

    nmf_results = None
    if compute_nmf:
        print("Computing Non-negative Matrix Factorization (NMF)...")
        nmf_results = NMF_calc(im, baseoutputfilename, output_dir, compat_options)
        print(f"  NMF completed")

    # Superpixel/voxel clustering
    print(f"Computing superpixel clustering (k={n_voxel_clusters})...")
    superpixels = voxel_cluster(im, compat_options)
    print(f"  Superpixel image shape: {superpixels.shape}")

    # Save superpixel visualization
    plot_img(
        superpixels, bestz, baseoutputfilename + "-Superpixels.png", output_dir, compat_options
    )

    # Channel PCA
    print(f"Computing channel PCA (n_components={n_channel_pca_components})...")
    reducedim = clusterchannels(
        im, baseoutputfilename, output_dir, interior_cells, compat_options
    )
    print(f"  PCA reduced dimensions: {reducedim.shape}")

    # Visualize top 3 PCA components
    PCA_img = plotprincomp(
        reducedim,
        bestz,
        baseoutputfilename + "-Top3ChannelPCA.png",
        output_dir,
        compat_options,
    )

    # Generate visualization OME-TIFF
    if generate_visualization_tiff:
        print("Generating visualization OME-TIFF...")
        write_ometiff(im, output_dir, compat_options, PCA_img, superpixels[bestz])
        print(f"  Visualization TIFF saved")

    # Create ImageAnalysisData object
    image_analysis_data = ImageAnalysisData(
        nmf_results=nmf_results,
        superpixels=superpixels,
        pca_components=reducedim,
        pca_img=PCA_img,
    )

    # Save checkpoint
    print("Saving image analysis checkpoint...")
    CheckpointManager.save_image_analysis_data(image_analysis_data, output_dir)

    print("✓ Image-level analysis complete")
    print("=" * 60)

    return image_analysis_data


def load_checkpoint(checkpoint_dir: Union[Path, str]) -> ImageAnalysisData:
    """
    Load previously saved image analysis checkpoint.

    Parameters:
    -----------
    checkpoint_dir : Path or str
        Directory containing checkpoints/ subdirectory

    Returns:
    --------
    ImageAnalysisData: Loaded image analysis data
    """
    checkpoint_dir = Path(checkpoint_dir)
    return CheckpointManager.load_image_analysis_data(
        CheckpointManager.get_checkpoint_dir(checkpoint_dir)
    )

