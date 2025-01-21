"""
    File for orchestrating the entire process of downscaling an image.
"""

# Standard Imports
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# custom imports
from pygan_downscaler.utils import toolbox as tb
from pygan_downscaler.utils import preprocessor as pp
from pygan_downscaler.utils import grid_gen as gg
from pygan_downscaler.utils import file_handler as fh

# location of the data files
DATA_PATH = "data/"

# get the list of files in the data folder
data_files = fh.get_filenames(DATA_PATH)

# load the data into this dictionary for easier referencing
datasets = {}

if __name__ == "__main__":

    # populate the datasets dictionary
    for name in data_files:
        # load the data into this module
        datasets[name] = fh.load(DATA_PATH, name)

    # start preprocessing the data
    # Step 1: select the long and lat slices
    for key, data in datasets.items():
        # select the longitude and latitude slices
        datasets[key] = pp.select_long_lat(
            data,
            long = [102.5, 105.00],
            lat = [2.5, 0.75]
            )

    # Step 2: remove duplicate time stamps
    for key, data in datasets.items():
        # remove duplicate time stamps
        datasets[key] = pp.remove_duplicates(data)

    split_data = {}

    # Step 3: break data into training, validation, and testing sets
    for key, data in datasets.items():
        # split the data into training, validation, and testing sets
        split_data[key] = pp.split_data(data)

    # step 4: remove extra fluff data (history and documentation)
    # will slow down data processing and are unnecessary
    