"""
Module 1: Core Preprocessing

Loads images, extracts ROI coordinates, performs quality control.
This module is required for all other modules.
"""

from pathlib import Path
from typing import Dict, Optional, Union

from ..data_structures import IMGstruct, MaskStruct
from ..SPRM_pkg import find_edge_cells, get_coordinates, quality_control
from .checkpoint_manager import CheckpointManager, CoreData


def run(
    img_file: Union[Path, str],
    mask_file: Union[Path, str],
    output_dir: Union[Path, str],
    image_dimension: str = "2D",
    debug: bool = False,
) -> CoreData:
    """
    Run core preprocessing: load images, extract ROIs, perform quality control.

    This module must run first as all other modules depend on its output.

    Parameters:
    -----------
    img_file : Path or str
        Path to OME-TIFF image file
    mask_file : Path or str
        Path to OME-TIFF mask/segmentation file
    output_dir : Path or str
        Directory for outputs and checkpoints
    image_dimension : str, default "2D"
        Image dimensionality: "2D" or "3D"
    debug : bool, default False
        Enable debug output

    Returns:
    --------
    CoreData: Preprocessed data including IMGstruct, MaskStruct, ROI coords, cell lists

    Checkpoints saved:
    ------------------
    - checkpoints/core_data.pkl
    - checkpoints/roi_coords.h5
    - checkpoints/cell_lists.json
    """
    img_file = Path(img_file)
    mask_file = Path(mask_file)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if checkpoint already exists
    if CheckpointManager.exists_core_data(output_dir):
        print(f"Loading existing preprocessing checkpoint from {output_dir}")
        return CheckpointManager.load_core_data(
            CheckpointManager.get_checkpoint_dir(output_dir)
        )

    print("=" * 60)
    print("SPRM Module 1: Core Preprocessing")
    print("=" * 60)

    # Create minimal options dict for compatibility with existing functions
    options = {
        "image_dimension": image_dimension,
        "debug": debug,
        "interior_cells_only": 1,  # Default behavior
        "valid_cell_threshold": 10,  # Minimum valid cells
    }

    # Load image and mask
    print(f"Reading image file: {img_file.name}")
    im = IMGstruct(img_file, options)

    if debug:
        print(f"Image dimensions: {im.get_data().shape}")

    print(f"Reading mask file: {mask_file.name}")
    mask = MaskStruct(mask_file, options)

    if debug:
        print(f"Mask dimensions: {mask.get_data().shape}")

    # Extract ROI coordinates for all channels in mask
    print("Extracting cell ROI coordinates...")
    ROI_coords = get_coordinates(mask, options)
    mask.set_ROI(ROI_coords)

    # Quality control: identify edge cells, bad cells, best z-planes
    print("Performing quality control...")
    quality_control(mask, im, ROI_coords, options)

    # Get cell lists
    interior_cells = mask.get_interior_cells()
    edge_cells = mask.get_edge_cells()
    cell_index = mask.get_cell_index()
    bad_cells = mask.get_bad_cells()
    bestz = mask.get_bestz()

    print(f"Found {len(cell_index)} total cells")
    print(f"  Interior cells: {len(interior_cells)}")
    print(f"  Edge cells: {len(edge_cells)}")
    print(f"  Bad cells: {len(bad_cells)}")
    print(f"  Best Z-plane(s): {bestz}")

    # Create CoreData object
    core_data = CoreData(
        im=im,
        mask=mask,
        roi_coords=ROI_coords,
        interior_cells=interior_cells,
        edge_cells=edge_cells,
        cell_index=cell_index,
        bad_cells=bad_cells,
        bestz=bestz,
    )

    # Save checkpoint
    print("Saving preprocessing checkpoint...")
    CheckpointManager.save_core_data(core_data, output_dir)

    print("✓ Preprocessing complete")
    print("=" * 60)

    return core_data


def load_checkpoint(checkpoint_dir: Union[Path, str]) -> CoreData:
    """
    Load previously saved preprocessing checkpoint.

    Parameters:
    -----------
    checkpoint_dir : Path or str
        Directory containing checkpoints/ subdirectory

    Returns:
    --------
    CoreData: Loaded preprocessing data
    """
    checkpoint_dir = Path(checkpoint_dir)
    return CheckpointManager.load_core_data(
        CheckpointManager.get_checkpoint_dir(checkpoint_dir)
    )

