import multiprocessing
import re

from frozendict import frozendict

INTEGER_PATTERN = re.compile(r'(\d+)')
FILENAMES_TO_IGNORE = frozenset({'.DS_Store'})
num_cores = multiprocessing.cpu_count()

figure_save_params = frozendict({'bbox_inches': 'tight', 'dpi': 300})
