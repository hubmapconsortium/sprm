"""
SPRM Modular API

This package provides modular access to SPRM pipeline components.
Each module can be run independently with checkpoint support.

Example usage:
    from sprm import modules

    # Run preprocessing
    core = modules.preprocessing.run(
        img_file="image.ome.tiff",
        mask_file="mask.ome.tiff",
        output_dir="sprm_outputs"
    )

    # Run feature extraction
    features = modules.cell_features.run(
        core_data=core,
        output_dir="sprm_outputs"
    )

    # Run clustering
    clusters = modules.clustering.run(
        core_data=core,
        cell_features=features,
        output_dir="sprm_outputs"
    )
"""

from . import (
    cell_features,
    checkpoint_manager,
    clustering,
    image_analysis,
    preprocessing,
    segmentation_eval,
    shape_analysis,
    spatial_graphs,
)
from .checkpoint_manager import (
    CellFeatures,
    CheckpointManager,
    ClusteringResults,
    CoreData,
    ImageAnalysisData,
    SegmentationMetrics,
    ShapeData,
    SpatialData,
)

__all__ = [
    # Modules
    "preprocessing",
    "segmentation_eval",
    "shape_analysis",
    "spatial_graphs",
    "image_analysis",
    "cell_features",
    "clustering",
    # Data classes
    "CoreData",
    "SegmentationMetrics",
    "ShapeData",
    "SpatialData",
    "ImageAnalysisData",
    "CellFeatures",
    "ClusteringResults",
    # Utilities
    "CheckpointManager",
    "checkpoint_manager",
]

