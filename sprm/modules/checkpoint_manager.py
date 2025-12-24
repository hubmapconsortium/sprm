"""
Checkpoint management utilities for SPRM modular pipeline.

Handles loading, saving, and auto-detection of checkpoint data between modules.
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import h5py
import numpy as np

from ..SPRM_pkg import compute_cell_centers
from ..data_structures import IMGstruct, MaskStruct


@dataclass
class CoreData:
    """Output from preprocessing module."""

    im: IMGstruct
    mask: MaskStruct
    roi_coords: List
    interior_cells: List[int]
    edge_cells: List[int]
    cell_index: List[int]
    bad_cells: Set[int]
    bestz: List[int]
    cell_centers: Optional[np.ndarray]


@dataclass
class SegmentationMetrics:
    """Output from segmentation_eval module."""

    metrics: Dict[str, Any]
    quality_score: Optional[float] = None


@dataclass
class ShapeData:
    """Output from shape_analysis module."""

    shape_vectors: np.ndarray
    norm_shape_vectors: np.ndarray
    outline_vectors: Optional[np.ndarray] = None
    cell_polygons: Optional[List] = None


@dataclass
class SpatialData:
    """Output from spatial_graphs module."""

    adjacency_matrix: np.ndarray
    cell_graph: Dict
    snr_data: Dict[str, float]


@dataclass
class ImageAnalysisData:
    """Output from image_analysis module."""

    nmf_results: Optional[Dict] = None
    superpixels: Optional[np.ndarray] = None
    pca_components: Optional[np.ndarray] = None
    pca_img: Optional[np.ndarray] = None


@dataclass
class CellFeatures:
    """Output from cell_features module."""

    mean_vector: np.ndarray
    covar_matrix: np.ndarray
    total_vector: np.ndarray
    texture_vectors: np.ndarray
    texture_channels: List[str]


@dataclass
class ClusteringResults:
    """Output from clustering module."""

    cluster_assignments: Dict[str, np.ndarray]
    cluster_centers: Dict[str, np.ndarray]
    cluster_scores: Optional[np.ndarray] = None
    celltype_assignments: Optional[Any] = None


class CheckpointManager:
    """Manages checkpoint operations for SPRM modules."""

    CHECKPOINT_DIR = "checkpoints"

    @staticmethod
    def get_checkpoint_dir(output_dir: Path) -> Path:
        """Get or create checkpoint directory within output_dir."""
        checkpoint_dir = Path(output_dir) / CheckpointManager.CHECKPOINT_DIR
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    @staticmethod
    def auto_load_core_data(
        data: Union[Path, str, CoreData, None],
        output_dir: Optional[Path] = None,
    ) -> CoreData:
        """
        Auto-detect and load CoreData from various input types.

        Parameters:
        -----------
        data : Path, str, CoreData, or None
            - If Path/str: Load checkpoint from this directory
            - If CoreData: Return as-is
            - If None: Try to load from output_dir/checkpoints/

        output_dir : Path, optional
            Fallback directory to look for checkpoints

        Returns:
        --------
        CoreData: Loaded or passed-through core data
        """
        if isinstance(data, CoreData):
            return data

        if data is None and output_dir is None:
            raise ValueError(
                "Either data or output_dir must be provided to load core data"
            )

        checkpoint_dir = (
            CheckpointManager.get_checkpoint_dir(Path(data) if data else output_dir)
        )

        return CheckpointManager.load_core_data(checkpoint_dir)

    @staticmethod
    def save_core_data(data: CoreData, output_dir: Path):
        """Save CoreData checkpoint."""
        checkpoint_dir = CheckpointManager.get_checkpoint_dir(output_dir)

        # Save pickle for IMGstruct and MaskStruct
        with open(checkpoint_dir / "core_data.pkl", "wb") as f:
            pickle.dump(
                {
                    "im": data.im,
                    "mask": data.mask,
                    "bestz": data.bestz,
                    "cell_centers": data.cell_centers,
                },
                f,
            )

        # Save HDF5 for ROI coords (can be large)
        with h5py.File(checkpoint_dir / "roi_coords.h5", "w") as f:
            for i, coords in enumerate(data.roi_coords):
                if coords is not None and len(coords) > 0:
                    # Each coords is a list of cell coordinates
                    for j, cell_coords in enumerate(coords):
                        if cell_coords is not None and len(cell_coords) > 0:
                            f.create_dataset(
                                f"channel_{i}/cell_{j}",
                                data=np.array(cell_coords),
                                compression="gzip",
                            )

        # Save cell lists as JSON (convert numpy types to Python types)
        def convert_to_native(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, set):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return convert_to_native(obj.tolist())
            return obj

        with open(checkpoint_dir / "cell_lists.json", "w") as f:
            json.dump(
                {
                    "interior_cells": convert_to_native(data.interior_cells),
                    "edge_cells": convert_to_native(data.edge_cells),
                    "cell_index": convert_to_native(data.cell_index),
                    "bad_cells": convert_to_native(list(data.bad_cells)),
                },
                f,
            )

    @staticmethod
    def load_core_data(checkpoint_dir: Path) -> CoreData:
        """Load CoreData checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        # Load pickle
        with open(checkpoint_dir / "core_data.pkl", "rb") as f:
            core_dict = pickle.load(f)

        # Load ROI coords from HDF5
        roi_coords = []
        with h5py.File(checkpoint_dir / "roi_coords.h5", "r") as f:
            # Determine number of channels
            n_channels = len([k for k in f.keys() if k.startswith("channel_")])

            for i in range(n_channels):
                channel_key = f"channel_{i}"
                if channel_key in f:
                    channel_group = f[channel_key]
                    n_cells = len([k for k in channel_group.keys()])

                    cells = []
                    for j in range(n_cells):
                        cell_key = f"cell_{j}"
                        if cell_key in channel_group:
                            cells.append(channel_group[cell_key][()])
                        else:
                            cells.append(None)
                    roi_coords.append(cells)
                else:
                    roi_coords.append([])

        # Load cell lists
        with open(checkpoint_dir / "cell_lists.json", "r") as f:
            cell_lists = json.load(f)

        cell_centers = core_dict.get("cell_centers")
        if cell_centers is None:
            cell_centers = compute_cell_centers(roi_coords)

        return CoreData(
            im=core_dict["im"],
            mask=core_dict["mask"],
            roi_coords=roi_coords,
            interior_cells=cell_lists["interior_cells"],
            edge_cells=cell_lists["edge_cells"],
            cell_index=cell_lists["cell_index"],
            bad_cells=set(cell_lists["bad_cells"]),
            bestz=core_dict["bestz"],
            cell_centers=cell_centers,
        )

    @staticmethod
    def exists_core_data(output_dir: Path) -> bool:
        """Check if CoreData checkpoint exists."""
        checkpoint_dir = CheckpointManager.get_checkpoint_dir(output_dir)
        return (
            (checkpoint_dir / "core_data.pkl").exists()
            and (checkpoint_dir / "roi_coords.h5").exists()
            and (checkpoint_dir / "cell_lists.json").exists()
        )


    @staticmethod
    def save_shape_data(data: ShapeData, output_dir: Path):
        """Save ShapeData checkpoint."""
        checkpoint_dir = CheckpointManager.get_checkpoint_dir(output_dir)

        with h5py.File(checkpoint_dir / "shape_features.h5", "w") as f:
            f.create_dataset(
                "shape_vectors", data=data.shape_vectors, compression="gzip"
            )
            f.create_dataset(
                "norm_shape_vectors", data=data.norm_shape_vectors, compression="gzip"
            )
            if data.outline_vectors is not None:
                f.create_dataset(
                    "outline_vectors", data=data.outline_vectors, compression="gzip"
                )

        # Save polygons as pickle if they exist
        if data.cell_polygons is not None:
            with open(checkpoint_dir / "cell_polygons.pkl", "wb") as f:
                pickle.dump(data.cell_polygons, f)

    @staticmethod
    def load_shape_data(checkpoint_dir: Path) -> ShapeData:
        """Load ShapeData checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        with h5py.File(checkpoint_dir / "shape_features.h5", "r") as f:
            shape_vectors = f["shape_vectors"][()]
            norm_shape_vectors = f["norm_shape_vectors"][()]
            outline_vectors = (
                f["outline_vectors"][()] if "outline_vectors" in f else None
            )

        cell_polygons = None
        if (checkpoint_dir / "cell_polygons.pkl").exists():
            with open(checkpoint_dir / "cell_polygons.pkl", "rb") as f:
                cell_polygons = pickle.load(f)

        return ShapeData(
            shape_vectors=shape_vectors,
            norm_shape_vectors=norm_shape_vectors,
            outline_vectors=outline_vectors,
            cell_polygons=cell_polygons,
        )

    @staticmethod
    def exists_shape_data(output_dir: Path) -> bool:
        """Check if ShapeData checkpoint exists."""
        checkpoint_dir = CheckpointManager.get_checkpoint_dir(output_dir)
        return (checkpoint_dir / "shape_features.h5").exists()

    @staticmethod
    def auto_load_shape_data(
        data: Union[Path, str, ShapeData, None],
        output_dir: Optional[Path] = None,
    ) -> Optional[ShapeData]:
        """Auto-detect and load ShapeData, returns None if not available."""
        if isinstance(data, ShapeData):
            return data

        if data is None and output_dir is None:
            return None

        checkpoint_dir = (
            CheckpointManager.get_checkpoint_dir(Path(data) if data else output_dir)
        )

        if not CheckpointManager.exists_shape_data(
            checkpoint_dir.parent
        ):  # parent because we get checkpoint dir
            return None

        return CheckpointManager.load_shape_data(checkpoint_dir)

    @staticmethod
    def save_spatial_data(data: SpatialData, output_dir: Path):
        """Save SpatialData checkpoint."""
        checkpoint_dir = CheckpointManager.get_checkpoint_dir(output_dir)

        with open(checkpoint_dir / "spatial_data.pkl", "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_spatial_data(checkpoint_dir: Path) -> SpatialData:
        """Load SpatialData checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        with open(checkpoint_dir / "spatial_data.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def exists_spatial_data(output_dir: Path) -> bool:
        """Check if SpatialData checkpoint exists."""
        checkpoint_dir = CheckpointManager.get_checkpoint_dir(output_dir)
        return (checkpoint_dir / "spatial_data.pkl").exists()

    @staticmethod
    def auto_load_spatial_data(
        data: Union[Path, str, SpatialData, None],
        output_dir: Optional[Path] = None,
    ) -> Optional[SpatialData]:
        """Auto-detect and load SpatialData, returns None if not available."""
        if isinstance(data, SpatialData):
            return data

        if data is None and output_dir is None:
            return None

        checkpoint_dir = (
            CheckpointManager.get_checkpoint_dir(Path(data) if data else output_dir)
        )

        if not CheckpointManager.exists_spatial_data(checkpoint_dir.parent):
            return None

        return CheckpointManager.load_spatial_data(checkpoint_dir)

    @staticmethod
    def save_image_analysis_data(data: ImageAnalysisData, output_dir: Path):
        """Save ImageAnalysisData checkpoint."""
        checkpoint_dir = CheckpointManager.get_checkpoint_dir(output_dir)

        with open(checkpoint_dir / "image_analysis.pkl", "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_image_analysis_data(checkpoint_dir: Path) -> ImageAnalysisData:
        """Load ImageAnalysisData checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        with open(checkpoint_dir / "image_analysis.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def exists_image_analysis_data(output_dir: Path) -> bool:
        """Check if ImageAnalysisData checkpoint exists."""
        checkpoint_dir = CheckpointManager.get_checkpoint_dir(output_dir)
        return (checkpoint_dir / "image_analysis.pkl").exists()

    @staticmethod
    def auto_load_image_analysis_data(
        data: Union[Path, str, ImageAnalysisData, None],
        output_dir: Optional[Path] = None,
    ) -> Optional[ImageAnalysisData]:
        """Auto-detect and load ImageAnalysisData, returns None if not available."""
        if isinstance(data, ImageAnalysisData):
            return data

        if data is None and output_dir is None:
            return None

        checkpoint_dir = (
            CheckpointManager.get_checkpoint_dir(Path(data) if data else output_dir)
        )

        if not CheckpointManager.exists_image_analysis_data(checkpoint_dir.parent):
            return None

        return CheckpointManager.load_image_analysis_data(checkpoint_dir)

    @staticmethod
    def save_cell_features(data: CellFeatures, output_dir: Path):
        """Save CellFeatures checkpoint."""
        checkpoint_dir = CheckpointManager.get_checkpoint_dir(output_dir)

        with h5py.File(checkpoint_dir / "cell_features.h5", "w") as f:
            f.create_dataset(
                "mean_vector", data=data.mean_vector, compression="gzip"
            )
            f.create_dataset(
                "covar_matrix", data=data.covar_matrix, compression="gzip"
            )
            f.create_dataset(
                "total_vector", data=data.total_vector, compression="gzip"
            )
            f.create_dataset(
                "texture_vectors", data=data.texture_vectors, compression="gzip"
            )

            # Save texture channels as attribute
            f.attrs["texture_channels"] = json.dumps(data.texture_channels)

    @staticmethod
    def load_cell_features(checkpoint_dir: Path) -> CellFeatures:
        """Load CellFeatures checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        with h5py.File(checkpoint_dir / "cell_features.h5", "r") as f:
            mean_vector = f["mean_vector"][()]
            covar_matrix = f["covar_matrix"][()]
            total_vector = f["total_vector"][()]
            texture_vectors = f["texture_vectors"][()]
            texture_channels = json.loads(f.attrs["texture_channels"])

        return CellFeatures(
            mean_vector=mean_vector,
            covar_matrix=covar_matrix,
            total_vector=total_vector,
            texture_vectors=texture_vectors,
            texture_channels=texture_channels,
        )

    @staticmethod
    def exists_cell_features(output_dir: Path) -> bool:
        """Check if CellFeatures checkpoint exists."""
        checkpoint_dir = CheckpointManager.get_checkpoint_dir(output_dir)
        return (checkpoint_dir / "cell_features.h5").exists()

    @staticmethod
    def auto_load_cell_features(
        data: Union[Path, str, CellFeatures, None],
        output_dir: Optional[Path] = None,
    ) -> CellFeatures:
        """Auto-detect and load CellFeatures."""
        if isinstance(data, CellFeatures):
            return data

        if data is None and output_dir is None:
            raise ValueError(
                "Either data or output_dir must be provided to load cell features"
            )

        checkpoint_dir = (
            CheckpointManager.get_checkpoint_dir(Path(data) if data else output_dir)
        )

        return CheckpointManager.load_cell_features(checkpoint_dir)

