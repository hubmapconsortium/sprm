"""
Module 1: Core Preprocessing

Loads images, extracts ROI coordinates, performs quality control.
This module is required for all other modules.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from ..data_structures import IMGstruct, MaskStruct
from ..SPRM_pkg import (
    compute_cell_centers,
    find_edge_cells,
    get_coordinates,
    quality_control,
    read_options,
)
from .checkpoint_manager import CheckpointManager, CoreData


def _ensure_cell_centers(core_data: CoreData, output_dir: Path):
    """
    Ensure `{image_name}-cell_centers.csv` exists for the provided core data.

    Parameters
    ----------
    core_data : CoreData
        Core preprocessing results containing ROI coordinates.
    output_dir : Path
        Directory where outputs should be written.
    """
    output_dir = Path(output_dir)
    base_name = core_data.im.get_name()
    centers_path = output_dir / f"{base_name}-cell_centers.csv"

    centers = core_data.cell_centers
    if centers is None:
        centers = compute_cell_centers(core_data.roi_coords)
        core_data.cell_centers = centers

    if centers is None or len(centers) == 0:
        print("Warning: Unable to write cell centers; no cell coordinate data available.")
        return

    if centers_path.exists():
        return

    centers_df = pd.DataFrame(centers, columns=["x", "y", "z"])
    centers_df.index.name = "ID"
    centers_df.to_csv(centers_path, index_label="ID")


def run(
    img_file: Union[Path, str],
    mask_file: Union[Path, str],
    output_dir: Union[Path, str],
    options: Optional[Union[Path, str, Dict]] = None,
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
    options : Path, str, dict, or None, default None
        Options for preprocessing. Can be:
        - Path/str to options.txt file (will be read)
        - Dictionary of options (already loaded)
        - None (use defaults with image_dimension and debug parameters)
    image_dimension : str, default "2D"
        Image dimensionality: "2D" or "3D" (only used if options is None)
    debug : bool, default False
        Enable debug output (only used if options is None)

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
        checkpoint_dir = CheckpointManager.get_checkpoint_dir(output_dir)
        core_data = CheckpointManager.load_core_data(checkpoint_dir)
        _ensure_cell_centers(core_data, output_dir)
        return core_data

    print("=" * 60)
    print("SPRM Module 1: Core Preprocessing")
    print("=" * 60)

    # Handle options parameter
    if options is None:
        # Create minimal options dict with defaults
        options = {
            "image_dimension": image_dimension,
            "debug": debug,
            "interior_cells_only": 1,  # Default behavior
            "valid_cell_threshold": 10,  # Minimum valid cells
        }
    elif isinstance(options, (str, Path)):
        # Read options from file
        print(f"Reading options from: {options}")
        options = dict(read_options(Path(options)))
        # Override with function parameters if provided
        if image_dimension != "2D":
            options["image_dimension"] = image_dimension
        if debug:
            options["debug"] = debug
    elif isinstance(options, dict):
        # Use provided options dict directly
        # Set defaults for required fields if not present
        if "image_dimension" not in options:
            options["image_dimension"] = image_dimension
        if "debug" not in options:
            options["debug"] = debug
        if "interior_cells_only" not in options:
            options["interior_cells_only"] = 1
        if "valid_cell_threshold" not in options:
            options["valid_cell_threshold"] = 10
    else:
        raise TypeError(
            f"options must be None, Path, str, or dict, got {type(options)}"
        )

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
    cell_centers = compute_cell_centers(ROI_coords)

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
        cell_centers=cell_centers,
    )

    _ensure_cell_centers(core_data, output_dir)

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
    core_data = CheckpointManager.load_core_data(
        CheckpointManager.get_checkpoint_dir(checkpoint_dir)
    )
    _ensure_cell_centers(core_data, checkpoint_dir)
    return core_data

