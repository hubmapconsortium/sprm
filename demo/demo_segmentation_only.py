#!/usr/bin/env python3
"""
Demo: Segmentation Evaluation Only

This script demonstrates running only segmentation quality evaluation
without any downstream analysis. Useful for quickly assessing
segmentation quality.
"""

from pathlib import Path

from sprm import modules

# Configuration
IMG_FILE = Path("img/image_demo.tiff")
MASK_FILE = Path("mask/mask_demo.tiff")
OUTPUT_DIR = Path("demo_segmentation_only_outputs")
OPTIONS_FILE = Path("../sprm/options.txt")

def main():
    """Run only segmentation evaluation."""

    print("\n" + "=" * 70)
    print("SPRM - Segmentation Evaluation Only")
    print("=" * 70 + "\n")

    # Step 1: Preprocessing (required)
    print("Step 1/2: Loading and preprocessing images")
    core = modules.preprocessing.run(
        img_file=IMG_FILE, mask_file=MASK_FILE, output_dir=OUTPUT_DIR, options=OPTIONS_FILE
    )

    # Step 2: Evaluate segmentation
    print("\nStep 2/2: Evaluating segmentation quality")
    seg_metrics = modules.segmentation_eval.run(
        core_data=core, output_dir=OUTPUT_DIR, standalone=True
    )

    print("\n" + "=" * 70)
    print("✓ Segmentation evaluation complete!")
    print(f"✓ Metrics saved to: {OUTPUT_DIR}")
    print("\nKey metrics:")
    for key, value in list(seg_metrics.metrics.items())[:10]:
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

