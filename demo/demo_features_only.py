#!/usr/bin/env python3
"""
Demo: Extract Mean and Total Features Only

This script demonstrates running only the preprocessing and feature extraction
modules to obtain mean and total intensity features without any downstream analysis.

Outputs:
- Mean intensity CSVs per channel (e.g., mean_features_ch01.csv)
- Total intensity CSVs per channel (e.g., total_features_ch01.csv)
- Covariance features (if multiple channels)
- Texture features (optional, can be disabled for speed)
"""

from pathlib import Path

from sprm import modules

# Configuration
IMG_FILE = Path("image.ome.tiff")
MASK_FILE = Path("mask.ome.tiff")
OUTPUT_DIR = Path("sprm_features_outputs")


def main():
    """Run SPRM pipeline to extract only mean and total features."""

    print("\n" + "=" * 70)
    print("SPRM Feature Extraction - Mean and Total Features")
    print("=" * 70 + "\n")

    # Step 1: Preprocessing
    print("Step 1/2: Core Preprocessing")
    print("Loading images and extracting ROIs...")
    core = modules.preprocessing.run(
        img_file=IMG_FILE,
        mask_file=MASK_FILE,
        output_dir=OUTPUT_DIR
    )
    print(f"✓ Extracted {core.num_rois} ROIs with {len(core.cell_lists)} cells")

    # Step 2: Cell Features (mean, total, covariance, optionally texture)
    print("\nStep 2/2: Cell Feature Extraction")
    print("Computing mean and total intensity features...")
    features = modules.cell_features.run(
        core_data=core,
        output_dir=OUTPUT_DIR,
        compute_texture=False,  # Set to True if you want texture features too
    )
    print(f"✓ Extracted features for {features.num_cells} cells")

    # Summary of outputs
    print("\n" + "=" * 70)
    print("✓ Feature extraction complete!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - mean_features_ch*.csv   (mean intensity per channel)")
    print("  - total_features_ch*.csv  (total intensity per channel)")
    print("  - covar_features.csv      (covariance features)")
    if features.texture_features is not None:
        print("  - texture_features_ch*.csv (texture features)")
    print("\nCheckpoint saved:")
    print(f"  - {OUTPUT_DIR}/checkpoints/cell_features.h5")
    print("\nYou can load features later with:")
    print('  features = modules.cell_features.load_checkpoint("sprm_features_outputs")')
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

