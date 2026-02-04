#!/usr/bin/env python3
"""
Demo: Illustrate loading features saved by demo_features_only.py

This script loads the features saved in the checkpoint folder and just
prints some basic info.

"""
from pathlib import Path
from sprm import modules

OUTPUT_DIR = Path("demo_features_only_outputs")

features = modules.cell_features.load_checkpoint(OUTPUT_DIR)
print("=======")
print("The checkpoint file contains a single class:")
print(type(features))
print("The methods and attributes of the class are:")
print(dir(features))
print("=======")
print("The mean vector is a numpy array:")
print(type(features.mean_vector))
print("Its shape is:")
print(features.mean_vector.shape)
print("=======")
