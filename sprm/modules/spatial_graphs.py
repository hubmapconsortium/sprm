"""
Module 4a: Spatial Graphs

Computes spatial relationships between cells including adjacency graphs and SNR.
This module is optional.
"""

from pathlib import Path
from typing import Union

from ..SPRM_pkg import SNR, cell_graphs
from .checkpoint_manager import CheckpointManager, CoreData, SpatialData


def run(
    core_data: Union[Path, str, CoreData],
    output_dir: Union[Path, str],
    adjacency_dilation_itr: int = 3,
    adjacency_delta: int = 3,
    compute_cell_graph: bool = True,
) -> SpatialData:
    """
    Compute spatial relationships: cell graphs, adjacency matrices, and SNR.

    Parameters:
    -----------
    core_data : Path, str, or CoreData
        Either a CoreData object or path to checkpoint directory
    output_dir : Path or str
        Directory for outputs and checkpoints
    adjacency_dilation_itr : int, default 3
        Number of dilation iterations for adjacency detection
    adjacency_delta : int, default 3
        Delta parameter for adjacency window calculation
    compute_cell_graph : bool, default True
        Whether to compute detailed cell graph structure

    Returns:
    --------
    SpatialData: Adjacency matrices, cell graphs, and SNR metrics

    Checkpoints saved:
    ------------------
    - checkpoints/spatial_data.pkl

    Outputs:
    --------
    - {image_name}-adjacency_matrix.csv
    - {image_name}-cell_graph.pkl
    - {image_name}-SNR.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-load core data if needed
    core_data = CheckpointManager.auto_load_core_data(core_data, output_dir)

    # Check if checkpoint already exists
    if CheckpointManager.exists_spatial_data(output_dir):
        print(f"Loading existing spatial analysis checkpoint from {output_dir}")
        return CheckpointManager.load_spatial_data(
            CheckpointManager.get_checkpoint_dir(output_dir)
        )

    print("=" * 60)
    print("SPRM Module 4a: Spatial Graphs")
    print("=" * 60)

    im = core_data.im
    mask = core_data.mask
    roi_coords = core_data.roi_coords
    interior_cells = core_data.interior_cells
    cellidx = core_data.cell_index
    baseoutputfilename = im.get_name()

    # Create options dict for compatibility
    options = {
        "cell_graph": 1 if compute_cell_graph else 0,
        "cell_adj_dilation_itr": adjacency_dilation_itr,
        "adj_matrix_delta": adjacency_delta,
        "debug": False,
    }

    # Compute cell graphs and adjacency
    if compute_cell_graph:
        print("Computing cell adjacency graphs...")
        adjacency_matrix, cell_graph = cell_graphs(
            mask, roi_coords, interior_cells, baseoutputfilename, output_dir, options
        )
        print(f"  Adjacency matrix shape: {adjacency_matrix.shape if hasattr(adjacency_matrix, 'shape') else 'N/A'}")
    else:
        print("Skipping cell graph computation")
        adjacency_matrix = None
        cell_graph = {}

    # Compute signal-to-noise ratio
    print("Computing signal-to-noise ratio per channel...")
    snr_data = SNR(im, baseoutputfilename, output_dir, cellidx, options)

    if snr_data is not None:
        print(f"  SNR computed for {len(snr_data)} channels")
        for channel, snr_value in list(snr_data.items())[:5]:  # Show first 5
            print(f"    {channel}: {snr_value:.4f}")
        if len(snr_data) > 5:
            print(f"    ... and {len(snr_data) - 5} more channels")
    else:
        snr_data = {}

    # Create SpatialData object
    spatial_data = SpatialData(
        adjacency_matrix=adjacency_matrix if adjacency_matrix is not None else {},
        cell_graph=cell_graph,
        snr_data=snr_data,
    )

    # Save checkpoint
    print("Saving spatial analysis checkpoint...")
    CheckpointManager.save_spatial_data(spatial_data, output_dir)

    print("✓ Spatial graph analysis complete")
    print("=" * 60)

    return spatial_data


def load_checkpoint(checkpoint_dir: Union[Path, str]) -> SpatialData:
    """
    Load previously saved spatial analysis checkpoint.

    Parameters:
    -----------
    checkpoint_dir : Path or str
        Directory containing checkpoints/ subdirectory

    Returns:
    --------
    SpatialData: Loaded spatial analysis data
    """
    checkpoint_dir = Path(checkpoint_dir)
    return CheckpointManager.load_spatial_data(
        CheckpointManager.get_checkpoint_dir(checkpoint_dir)
    )

