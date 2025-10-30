"""
Module 2: Segmentation Evaluation

Computes quality metrics for segmentation masks.
This module is optional and can run standalone.
"""

import json
from pathlib import Path
from typing import Union

from ..single_method_eval import single_method_eval
from ..single_method_eval_3D import single_method_eval_3D
from ..SPRM_pkg import NumpyEncoder
from .checkpoint_manager import CheckpointManager, CoreData, SegmentationMetrics


def run(
    core_data: Union[Path, str, CoreData],
    output_dir: Union[Path, str],
    image_dimension: str = "2D",
    standalone: bool = False,
) -> SegmentationMetrics:
    """
    Evaluate segmentation quality metrics.

    Parameters:
    -----------
    core_data : Path, str, or CoreData
        Either a CoreData object or path to checkpoint directory
    output_dir : Path or str
        Directory for outputs
    image_dimension : str, default "2D"
        Image dimensionality: "2D" or "3D"
    standalone : bool, default False
        If True, only compute segmentation metrics (no other modules will run)

    Returns:
    --------
    SegmentationMetrics: Computed metrics

    Outputs:
    --------
    - {image_name}-SPRM_Image_Quality_Measures.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-load core data if needed
    core_data = CheckpointManager.auto_load_core_data(core_data, output_dir)

    print("=" * 60)
    print("SPRM Module 2: Segmentation Evaluation")
    print("=" * 60)

    im = core_data.im
    mask = core_data.mask

    # Run appropriate evaluation based on dimension
    print(f"Computing segmentation metrics ({image_dimension})...")
    if image_dimension == "3D":
        seg_metrics = single_method_eval_3D(im, mask, output_dir)
    else:
        seg_metrics = single_method_eval(im, mask, output_dir)

    print("Segmentation metrics computed:")
    for key, value in seg_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save metrics to JSON
    metrics_file = output_dir / (im.name + "-SPRM_Image_Quality_Measures.json")
    struct = {"Segmentation Evaluation Metrics v1.5": seg_metrics}

    with open(metrics_file, "w") as json_file:
        json.dump(struct, json_file, indent=4, sort_keys=True, cls=NumpyEncoder)

    print(f"✓ Metrics saved to: {metrics_file.name}")

    if standalone:
        print("Standalone mode: Segmentation evaluation complete")
        print("=" * 60)

    result = SegmentationMetrics(metrics=seg_metrics)

    print("✓ Segmentation evaluation complete")
    print("=" * 60)

    return result

