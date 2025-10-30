"""
Module 4b: Image-Level Analysis

Performs whole-image analyses: NMF, superpixel clustering, channel PCA.
This module is optional.
"""

from pathlib import Path
from typing import Union

from ..SPRM_pkg import (
    NMF_calc,
    clusterchannels,
    plot_img,
    plotprincomp,
    voxel_cluster,
    write_ometiff,
)
from .checkpoint_manager import CheckpointManager, CoreData, ImageAnalysisData


def run(
    core_data: Union[Path, str, CoreData],
    output_dir: Union[Path, str],
    n_voxel_clusters: int = 3,
    n_channel_pca_components: int = 3,
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

    # Create options dict for compatibility
    options = {
        "num_voxelclusters": n_voxel_clusters,
        "num_channelPCA_components": n_channel_pca_components,
        "debug": False,
        "image_dimension": "2D",
    }

    nmf_results = None
    if compute_nmf:
        print("Computing Non-negative Matrix Factorization (NMF)...")
        nmf_results = NMF_calc(im, baseoutputfilename, output_dir, options)
        print(f"  NMF completed")

    # Superpixel/voxel clustering
    print(f"Computing superpixel clustering (k={n_voxel_clusters})...")
    superpixels = voxel_cluster(im, options)
    print(f"  Superpixel image shape: {superpixels.shape}")

    # Save superpixel visualization
    plot_img(
        superpixels, bestz, baseoutputfilename + "-Superpixels.png", output_dir, options
    )

    # Channel PCA
    print(f"Computing channel PCA (n_components={n_channel_pca_components})...")
    reducedim = clusterchannels(im, baseoutputfilename, output_dir, interior_cells, options)
    print(f"  PCA reduced dimensions: {reducedim.shape}")

    # Visualize top 3 PCA components
    PCA_img = plotprincomp(
        reducedim,
        bestz,
        baseoutputfilename + "-Top3ChannelPCA.png",
        output_dir,
        options,
    )

    # Generate visualization OME-TIFF
    if generate_visualization_tiff:
        print("Generating visualization OME-TIFF...")
        write_ometiff(im, output_dir, options, PCA_img, superpixels[bestz])
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

