#!/usr/bin/env python3
"""
Demo: Full SPRM Pipeline using Modular API

This script demonstrates running the complete SPRM pipeline using
the new modular interface.
"""

from pathlib import Path

from sprm import modules

# Configuration
IMG_FILE = Path("img/image_demo.tiff")
MASK_FILE = Path("mask/mask_demo.tiff")
OUTPUT_DIR = Path("demo_full_pipeline_outputs")
OPTIONS_FILE = Path("../sprm/options.txt")

def main():
    """Run complete SPRM pipeline with all modules."""

    print("\n" + "=" * 70)
    print("SPRM Modular Pipeline - Full Analysis")
    print("=" * 70 + "\n")

    # Module 1: Core Preprocessing (Required)
    print("Step 1/6: Core Preprocessing")
    core = modules.preprocessing.run(
        img_file=IMG_FILE, mask_file=MASK_FILE, output_dir=OUTPUT_DIR, options=OPTIONS_FILE
    )

    # Module 2: Segmentation Evaluation (Optional)
    print("\nStep 2/6: Segmentation Evaluation")
    seg_metrics = modules.segmentation_eval.run(
        core_data=core, output_dir=OUTPUT_DIR
    )

    # Module 3: Shape Analysis (Optional)
    print("\nStep 3/6: Shape Analysis")
    shape = modules.shape_analysis.run(
        core_data=core, output_dir=OUTPUT_DIR, n_outline_points=100
    )

    # Module 4a: Spatial Graphs (Optional)
    print("\nStep 4a/6: Spatial Graphs")
    spatial = modules.spatial_graphs.run(core_data=core, output_dir=OUTPUT_DIR)

    # Module 4b: Image-Level Analysis (Optional)
    print("\nStep 4b/6: Image-Level Analysis")
    image_analysis = modules.image_analysis.run(
        core_data=core,
        output_dir=OUTPUT_DIR,
        n_voxel_clusters=3,
        compute_nmf=True,
    )

    # Module 5: Cell Feature Extraction (Required for clustering)
    print("\nStep 5/6: Cell Feature Extraction")
    features = modules.cell_features.run(
        core_data=core,
        output_dir=OUTPUT_DIR,
        shape_data=shape,
        compute_texture=False,  # Can set to False to speed up
    )

    # Module 6: Clustering & Cell Typing
    print("\nStep 6/6: Clustering & Cell Typing")
    clusters = modules.clustering.run(
        core_data=core,
        cell_features=features,
        shape_data=shape,
        spatial_data=spatial,
        image_analysis_data=image_analysis,
        output_dir=OUTPUT_DIR,
        n_clusters_range=(3, 10),
        num_markers=3,
    )

    print("\n" + "=" * 70)
    print("✓ Full pipeline complete!")
    print(f"✓ All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

