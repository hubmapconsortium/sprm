#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

from sprm import SPRM

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('img_dir', type=Path)
    p.add_argument('mask_dir', type=Path)
    p.add_argument('--output-dir', type=Path, default=SPRM.DEFAULT_OUTPUT_PATH)
    p.add_argument('--options-file', type=Path, default=SPRM.DEFAULT_OPTIONS_FILE)
    p.add_argument('optional_img_dir', type=Path, nargs='?', default=False)
    argss = p.parse_args()

    if argss.optional_img_dir:
        SPRM.main(argss.img_dir, argss.mask_dir, argss.output_dir, argss.options_file, argss.optional_img_dir)
    else:
        SPRM.main(argss.img_dir, argss.mask_dir, argss.output_dir, argss.options_file)
