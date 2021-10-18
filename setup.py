from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent.absolute()

with open(here / "README.md", encoding="utf-8") as f:
    long_description = f.read()

with open(here / "sprm/version.txt") as f:
    version = f.read().strip()

setup(
    name="SPRM",
    version=version,
    description="Spatial Process & Relationship Modeling ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hubmapconsortium/sprm",
    author="Ted Zhang, Robert F. Murphy",
    author_email="tedz@andrew.cmu.edu, murphy@andrew.cmu.edu",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="sprm",
    packages=find_packages(),
    package_data={
        "": ["*.txt", "*.pickle"],
    },
    install_requires=[
        "aicsimageio<3.2",
        "frozendict",
        "manhole",
        "matplotlib",
        "numba",
        "numpy",
        "pandas",
        "pint",
        "scikit-image",
        "scikit-learn",
        "shapely",
        "tables",
        "tifffile==2020.2.16",
        "xmltodict",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "sprm=sprm.SPRM:argparse_wrapper",
        ],
    },
    zip_safe=False,
)
