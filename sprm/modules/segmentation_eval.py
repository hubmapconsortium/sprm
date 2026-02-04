"""
Module 2: Segmentation Evaluation

Computes quality metrics for segmentation masks.
This module is optional and can run standalone.
"""

import json
import importlib.resources
from contextlib import contextmanager
from pathlib import Path
from typing import Union

from ..single_method_eval import single_method_eval
from ..single_method_eval_3D import single_method_eval_3D
from ..SPRM_pkg import NumpyEncoder
from .checkpoint_manager import CheckpointManager, CoreData, SegmentationMetrics


@contextmanager
def _force_sprm_pickle_from_source_tree():
    """
    Work around an environment/path issue where a top-level `demo/sprm/` directory can
    cause `importlib.resources.open_binary("sprm", ...)` to resolve to the wrong place
    (namespace package portion), missing the PCA pickle resources.

    This temporarily intercepts `importlib.resources.open_binary` for the two PCA pickle
    resources and redirects them to the files shipped in this repo's `sprm/` package
    directory (adjacent to this module's parent package).
    """
    orig_open_binary = importlib.resources.open_binary
    pkg_dir = Path(__file__).resolve().parents[1]  # .../sprm/sprm

    def _open_binary_patched(package: str, resource: str):
        if package == "sprm" and resource in {"pca.pickle", "pca_3D.pickle"}:
            return (pkg_dir / resource).open("rb")
        return orig_open_binary(package, resource)

    importlib.resources.open_binary = _open_binary_patched  # type: ignore[assignment]
    try:
        yield
    finally:
        importlib.resources.open_binary = orig_open_binary  # type: ignore[assignment]


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
    with _force_sprm_pickle_from_source_tree():
        if image_dimension == "3D":
            eval_result = single_method_eval_3D(im, mask, output_dir)
        else:
            eval_result = single_method_eval(im, mask, output_dir)

    # The legacy evaluators return a tuple:
    # (metrics_dict, fraction_background, inv_background_cv, background_pca)
    # We primarily expose the metrics dict in the modular API.
    if isinstance(eval_result, tuple):
        seg_metrics = eval_result[0]
    else:
        seg_metrics = eval_result

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

    result = SegmentationMetrics(
        metrics=seg_metrics,
        quality_score=seg_metrics.get("QualityScore") if isinstance(seg_metrics, dict) else None,
    )

    print("✓ Segmentation evaluation complete")
    print("=" * 60)

    return result

