from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent.absolute()

with open(here / "README.md", encoding="utf-8") as f:
    long_description = f.read()

with open(here / "sprm/version.txt") as f:
    version = f.read().strip()

setup(
    name="sprmpkg",
    version=version,
    description="Spatial Process & Relationship Modeling ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hubmapconsortium/sprm",
    author="Ted Zhang, Robert F. Murphy",
    author_email="tedz@andrew.cmu.edu, murphy@andrew.cmu.edu",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="sprm",
    packages=find_packages(),
    package_data={
        "": ["*.txt", "*.pickle"],
    },
    install_requires=[
        "bioio",
        "bioio-ome-tiff",
        "bioio-tifffile",
        "anndata",
        "frozendict",
        "h5py",
        "lxml",
        "manhole",
        "matplotlib",
        "notebook",
        "numba",
        "numpy",
        "opencv-python",
        "pandas",
        "pint",
        "scikit-image",
        "scikit-learn",
        "shapely",
        "spatialdata",
        "tables",
        "tifffile",
        "xarray",
        "xmltodict",
        "umap-learn",
        "apng",
        "threadpoolctl>3",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "sprm=sprm.SPRM:argparse_wrapper",
        ],
    },
    zip_safe=False,
)
