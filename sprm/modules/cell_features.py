"""
Module 5: Cell Feature Extraction

Extracts per-cell features: intensity statistics, covariance, texture.
This module is required for clustering.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..SPRM_pkg import (
    build_matrix,
    build_vector,
    calculations,
    glcmProcedure,
    reallocate_and_merge_intensities,
    save_all,
)
from .checkpoint_manager import CheckpointManager, CellFeatures, CoreData, ShapeData


def run(
    core_data: Union[Path, str, CoreData],
    output_dir: Union[Path, str],
    shape_data: Optional[Union[Path, str, ShapeData]] = None,
    optional_img_file: Optional[Union[Path, str]] = None,
    compute_texture: bool = True,
    glcm_angles: list = None,
    glcm_distances: list = None,
) -> CellFeatures:
    """
    Extract per-cell features: mean, covariance, total intensity, and texture.

    Parameters:
    -----------
    core_data : Path, str, or CoreData
        Either a CoreData object or path to checkpoint directory
    output_dir : Path or str
        Directory for outputs and checkpoints
    shape_data : Path, str, ShapeData, or None
        Optional shape data to include in output CSVs
    optional_img_file : Path, str, or None
        Optional additional image file to merge with main image
    compute_texture : bool, default True
        Whether to compute GLCM texture features (can be slow)
    glcm_angles : list, optional
        Angles for GLCM computation (default: [0])
    glcm_distances : list, optional
        Distances for GLCM computation (default: [1])

    Returns:
    --------
    CellFeatures: Per-cell feature matrices

    Checkpoints saved:
    ------------------
    - checkpoints/cell_features.h5

    Outputs:
    --------
    - {image_name}-cell_channel_meanAll.csv
    - {image_name}-{channel}_channel_mean.csv (per mask channel)
    - {image_name}-{channel}_channel_covar.csv (per mask channel)
    - {image_name}-{channel}_channel_total.csv (per mask channel)
    - {image_name}-{channel}_texture.csv (per mask channel, if enabled)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-load core data if needed
    core_data = CheckpointManager.auto_load_core_data(core_data, output_dir)

    # Auto-load shape data if provided
    shape_data = CheckpointManager.auto_load_shape_data(shape_data, output_dir)

    # Check if checkpoint already exists
    if CheckpointManager.exists_cell_features(output_dir):
        print(f"Loading existing cell features checkpoint from {output_dir}")
        return CheckpointManager.load_cell_features(
            CheckpointManager.get_checkpoint_dir(output_dir)
        )

    print("=" * 60)
    print("SPRM Module 5: Cell Feature Extraction")
    print("=" * 60)

    im = core_data.im
    mask = core_data.mask
    roi_coords = core_data.roi_coords
    interior_cells = core_data.interior_cells
    cellidx = core_data.cell_index
    bestz = core_data.bestz
    baseoutputfilename = im.get_name()

    # Create options dict for compatibility
    if glcm_angles is None:
        glcm_angles = [0]
    if glcm_distances is None:
        glcm_distances = [1]

    options = {
        "skip_texture": 0 if compute_texture else 1,
        "glcm_angles": glcm_angles,
        "glcm_distances": glcm_distances,
        "debug": False,
        "image_dimension": "2D",
    }

    # Reallocate intensities to match mask resolution
    print("Reallocating intensities to mask resolution...")
    opt_img_file = Path(optional_img_file) if optional_img_file else None
    reallocate_and_merge_intensities(im, mask, opt_img_file, options)
    print(f"  Image data shape after reallocation: {im.get_data().shape}")

    # Compute texture features
    if compute_texture:
        print("Computing GLCM texture features...")
        print(f"  Angles: {glcm_angles}, Distances: {glcm_distances}")
        textures = glcmProcedure(
            im, mask, output_dir, baseoutputfilename, roi_coords, options
        )
        print(f"  Texture matrix shape: {textures[0].shape}")
    else:
        print("Skipping texture computation (using zeros)")
        # Create fake texture matrix with zeros
        cell_count = len(interior_cells)
        textures = [
            np.zeros((1, 2, cell_count, len(im.channel_labels) * 6, 1)),
            im.channel_labels * 12,
        ]
        # Save empty texture files
        for i in range(2):
            df = pd.DataFrame(
                textures[0][0, i, :, :, 0],
                columns=textures[1][: len(im.channel_labels) * 6],
                index=list(range(1, len(interior_cells) + 1)),
            )
            df.index.name = "ID"
            df.to_csv(
                output_dir
                / (baseoutputfilename + "-" + mask.channel_labels[i] + "_1_texture.csv")
            )

    # Initialize feature matrices
    print("Computing per-cell intensity statistics...")
    covar_matrix = []
    mean_vector = []
    total_vector = []

    # Loop over time points (usually just 1)
    for t in range(0, im.get_data().shape[1]):
        # Loop over mask channels (e.g., cell, nucleus, membranes)
        for j in range(0, mask.get_data().shape[2]):
            masked_imgs_coord = roi_coords[j]
            # Filter to only interior cells
            masked_imgs_coord = [masked_imgs_coord[i] for i in interior_cells]

            # Build matrices for this mask channel
            covar_matrix = build_matrix(im, mask, masked_imgs_coord, j, covar_matrix)
            mean_vector = build_vector(im, mask, masked_imgs_coord, j, mean_vector)
            total_vector = build_vector(im, mask, masked_imgs_coord, j, total_vector)

            # Extract ROI pixel intensity matrices for all cells in this mask channel.
            # `calculations` returns {cell_idx -> ROI} where ROI has shape (C, Npix).
            roi_dict = calculations(masked_imgs_coord, im, t, bestz)

            # Compute statistics for each cell
            for i in range(0, len(masked_imgs_coord)):
                roi = roi_dict.get(i)
                c = im.img.dims.C
                if roi is None or getattr(roi, "size", 0) == 0:
                    covar_matrix[t, j, i, :, :] = np.zeros((c, c), dtype=np.float64)
                    mean_vector[t, j, i, :, :] = np.zeros((c, 1), dtype=np.float64)
                    total_vector[t, j, i, :, :] = np.zeros((c, 1), dtype=np.float64)
                    continue

                # Mean / total per channel
                mean_vector[t, j, i, :, :] = roi.mean(axis=1, keepdims=True)
                total_vector[t, j, i, :, :] = roi.sum(axis=1, keepdims=True)

                # Covariance between channels
                if roi.shape[1] > 1:
                    cov = np.cov(roi)
                    if cov.ndim == 0:
                        cov = np.array([[float(cov)]], dtype=np.float64)
                    covar_matrix[t, j, i, :, :] = np.nan_to_num(
                        cov, nan=0.0, posinf=0.0, neginf=0.0
                    )
                else:
                    covar_matrix[t, j, i, :, :] = np.zeros((c, c), dtype=np.float64)

    print(f"  Mean vector shape: {mean_vector.shape}")
    print(f"  Covariance matrix shape: {covar_matrix.shape}")
    print(f"  Total vector shape: {total_vector.shape}")

    # Get shape vectors if available
    shape_vectors = shape_data.shape_vectors if shape_data else None
    norm_shape_vectors = shape_data.norm_shape_vectors if shape_data else None

    # Save all features to CSV files
    print("Writing feature matrices to CSV files...")
    save_all(
        filename=baseoutputfilename,
        im=im,
        mask=mask,
        output_dir=output_dir,
        cellidx=cellidx,
        options=options,
        mean_vector=mean_vector,
        covar_matrix=covar_matrix,
        total_vector=total_vector,
        outline_vectors=shape_vectors,
        norm_shape_vectors=norm_shape_vectors,
    )

    # Create CellFeatures object
    cell_features = CellFeatures(
        mean_vector=mean_vector,
        covar_matrix=covar_matrix,
        total_vector=total_vector,
        texture_vectors=textures[0],
        texture_channels=textures[1],
    )

    # Save checkpoint
    print("Saving cell features checkpoint...")
    CheckpointManager.save_cell_features(cell_features, output_dir)

    print("✓ Cell feature extraction complete")
    print("=" * 60)

    return cell_features


def load_checkpoint(checkpoint_dir: Union[Path, str]) -> CellFeatures:
    """
    Load previously saved cell features checkpoint.

    Parameters:
    -----------
    checkpoint_dir : Path or str
        Directory containing checkpoints/ subdirectory

    Returns:
    --------
    CellFeatures: Loaded cell features
    """
    checkpoint_dir = Path(checkpoint_dir)
    return CheckpointManager.load_cell_features(
        CheckpointManager.get_checkpoint_dir(checkpoint_dir)
    )

