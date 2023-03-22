# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:40:13 2023

@author: jpv88
"""

from pathlib import Path
import matplotlib.pyplot as plt

import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache

# Confirming your allensdk version
print(f"Your allensdk version is: {allensdk.__version__}")

# Update this to a valid directory in your filesystem. This is where the data will be stored.
output_dir = "C:/Users/jpv88/Downloads"
DOWNLOAD_COMPLETE_DATASET = False

output_dir = Path(output_dir)
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=output_dir)

# %%

units = cache.get_unit_table()

print(f"Total number of units: {len(units)}")

units.head()