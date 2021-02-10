from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent.absolute()

with open(here / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='SPRM',
    version='0.55',
    description='Spatial Process & Relationship Modeling ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hubmapconsortium/sprm',
    author='Ted Zhang, Robert F. Murphy',
    author_email='tedz@andrew.cmu.edu, murphy@andrew.cmu.edu',
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='sprm',
    packages=find_packages(),
    package_data={"": ["*.txt"],},
    install_requires=[
        'aicsimageio',
        'manhole',
        'matplotlib',
        'numba',
        'numpy',
        'pandas',
        'scikit-image',
        'scikit-learn',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'sprm=sprm.SPRM:argparse_wrapper',
        ],
    },
    zip_safe=False,
)
