from os import walk
from pathlib import Path
from typing import Sequence, Union

from .constants import FILENAMES_TO_IGNORE, INTEGER_PATTERN


def try_parse_int(value: str) -> Union[int, str]:
    if value.isdigit():
        return int(value)
    return value


def alphanum_sort_key(path: Path) -> Sequence[Union[int, str]]:
    """
    By: Matt Ruffalo
    Produces a sort key for file names, alternating strings and integers.
    Always [string, (integer, string)+] in quasi-regex notation.
    >>> alphanum_sort_key(Path('s1 1 t.tiff'))
    ['s', 1, ' ', 1, ' t.tiff']
    >>> alphanum_sort_key(Path('0_4_reg001'))
    ['', 0, '_', 4, '_reg', 1, '']
    """
    return [try_parse_int(c) for c in INTEGER_PATTERN.split(path.name)]


def get_paths(img_dir: Path) -> list[Path]:
    if img_dir.is_dir():
        img_files = []

        for dirpath_str, _, filenames in walk(img_dir):
            dirpath = Path(dirpath_str)
            filenames_usable = set(filenames) - FILENAMES_TO_IGNORE
            img_files.extend(dirpath / filename for filename in filenames_usable)
    else:
        # assume it's a pattern, like Path('some/dir/*.tiff')
        # don't need to filter filenames, because the user passed a
        # glob pattern of exactly what is wanted
        img_files = list(img_dir.parent.glob(img_dir.name))

    return sorted(img_files, key=alphanum_sort_key)
