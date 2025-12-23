#!/usr/bin/env python3
"""
Demo: Minimal SPRM Pipeline

This script demonstrates running only the essential modules:
1. Preprocessing
2. Cell Feature Extraction
3. Clustering

This is faster as it skips shape analysis, spatial graphs, and image-level analysis.
"""

from pathlib import Path

from sprm import modules

# Configuration
IMG_FILE = Path("img/image_demo.tiff")
MASK_FILE = Path("mask/mask_demo.tiff")
OUTPUT_DIR = Path("sprm_minimal_outputs")
OPTIONS_FILE = Path("/hive/users/tedz/workspace/sprm/sprm/options.txt")


def main():
    """Run minimal SPRM pipeline - only essential modules."""

    print("\n" + "=" * 70)
    print("SPRM Modular Pipeline - Minimal Analysis (Fast)")
    print("=" * 70 + "\n")

    # Step 1: Preprocessing
    print("Step 1/3: Core Preprocessing")
    core = modules.preprocessing.run(
        img_file=IMG_FILE, mask_file=MASK_FILE, output_dir=OUTPUT_DIR, options=OPTIONS_FILE
    )

    # Step 2: Cell Features (skip texture for speed)
    print("\nStep 2/3: Cell Feature Extraction (no texture)")
    features = modules.cell_features.run(
        core_data=core,
        output_dir=OUTPUT_DIR,
        compute_texture=False,  # Skip texture to save time
    )

    # Step 3: Clustering
    print("\nStep 3/3: Clustering")
    clusters = modules.clustering.run(
        core_data=core,
        cell_features=features,
        output_dir=OUTPUT_DIR,
        n_clusters_range=(3, 10),
        num_markers=3,  # Set the number of markers to 3
    )

    print("\n" + "=" * 70)
    print("✓ Minimal pipeline complete!")
    print(f"✓ All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

