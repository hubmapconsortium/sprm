"""
Module 3: Shape Analysis

Extracts cell shape features using parametric outlines and PCA.
This module is optional.
"""

from pathlib import Path
from typing import Union

from ..outlinePCA import (
    bin_pca,
    get_parametric_outline,
    getcellshapefeatures,
    kmeans_cluster_shape,
    pca_recon,
)
from ..SPRM_pkg import write_cell_polygs
from .checkpoint_manager import CheckpointManager, CoreData, ShapeData


def run(
    core_data: Union[Path, str, CoreData],
    output_dir: Union[Path, str],
    n_outline_points: int = 100,
    debug: bool = False,
) -> ShapeData:
    """
    Extract cell shape features using parametric outlines and PCA.

    Parameters:
    -----------
    core_data : Path, str, or CoreData
        Either a CoreData object or path to checkpoint directory
    output_dir : Path or str
        Directory for outputs and checkpoints
    n_outline_points : int, default 100
        Number of points for parametric outline representation
    debug : bool, default False
        Enable debug visualizations (PCA reconstructions, etc.)

    Returns:
    --------
    ShapeData: Shape vectors and cell polygons

    Checkpoints saved:
    ------------------
    - checkpoints/shape_features.h5
    - checkpoints/cell_polygons.pkl

    Outputs:
    --------
    - {image_name}-cell_outline.csv
    - {image_name}-cell_polygons.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-load core data if needed
    core_data = CheckpointManager.auto_load_core_data(core_data, output_dir)

    # Check if checkpoint already exists
    if CheckpointManager.exists_shape_data(output_dir):
        print(f"Loading existing shape analysis checkpoint from {output_dir}")
        return CheckpointManager.load_shape_data(
            CheckpointManager.get_checkpoint_dir(output_dir)
        )

    print("=" * 60)
    print("SPRM Module 3: Shape Analysis")
    print("=" * 60)

    mask = core_data.mask
    roi_coords = core_data.roi_coords
    cellidx = core_data.cell_index

    # Get cell segmentation channel
    seg_n = mask.get_labels("cell")

    # Create options dict for compatibility
    options = {
        "num_outlinepoints": n_outline_points,
        "debug": debug,
        "image_dimension": "2D",  # Shape analysis primarily for 2D
    }

    # Extract parametric outlines
    print(f"Extracting parametric cell outlines ({n_outline_points} points)...")
    outline_vectors, cell_polygons = get_parametric_outline(
        mask,
        seg_n,
        roi_coords,
        options,
    )

    print(f"Computing shape features using PCA...")
    shape_vectors, norm_shape_vectors, pca = getcellshapefeatures(
        outline_vectors, options
    )

    print(f"Shape feature dimensions: {shape_vectors.shape}")
    print(f"Normalized shape feature dimensions: {norm_shape_vectors.shape}")

    # Debug visualizations if requested
    if debug:
        print("Generating debug visualizations...")
        baseoutputfilename = core_data.im.get_name()

        # K-means clustering on shape
        kmeans_cluster_shape(shape_vectors, outline_vectors, output_dir, options)

        # PCA binning visualization
        bin_pca(norm_shape_vectors, 1, outline_vectors, baseoutputfilename, output_dir)

        # PCA reconstruction
        pca_recon(norm_shape_vectors, 1, pca, baseoutputfilename, output_dir)

    # Write cell polygons to CSV
    print("Writing cell polygons and outlines to CSV...")
    baseoutputfilename = core_data.im.get_name()
    write_cell_polygs(
        cell_polygons,
        outline_vectors,
        cellidx,
        baseoutputfilename,
        output_dir,
        options,
    )

    # Create ShapeData object
    shape_data = ShapeData(
        shape_vectors=shape_vectors,
        norm_shape_vectors=norm_shape_vectors,
        outline_vectors=outline_vectors,
        cell_polygons=cell_polygons,
    )

    # Save checkpoint
    print("Saving shape analysis checkpoint...")
    CheckpointManager.save_shape_data(shape_data, output_dir)

    print("✓ Shape analysis complete")
    print("=" * 60)

    return shape_data


def load_checkpoint(checkpoint_dir: Union[Path, str]) -> ShapeData:
    """
    Load previously saved shape analysis checkpoint.

    Parameters:
    -----------
    checkpoint_dir : Path or str
        Directory containing checkpoints/ subdirectory

    Returns:
    --------
    ShapeData: Loaded shape analysis data
    """
    checkpoint_dir = Path(checkpoint_dir)
    return CheckpointManager.load_shape_data(
        CheckpointManager.get_checkpoint_dir(checkpoint_dir)
    )

