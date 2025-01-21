import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl

def select_long_lat(data: xr.Dataset, long=None, lat=None) -> xr.Dataset:
    """
    Select the longitude and latitude slices from the data.

    Args:
        data (xarray dataset): data to slice
        long (list): list of two floats representing the longitude slice
        lat (list): list of two floats representing the latitude slice

    Returns:
        xarray dataset: sliced data
    """
    # slice the data if there is a slice range provided, else do nothing
    return data.sel(
        lon=slice(long[0], long[1]),
        lat=slice(lat[0], lat[1])
        ) if long and lat else data

def remove_duplicates(data: xr.Dataset, dimension="time") -> xr.Dataset:
    """
    Remove duplicate dimensional value (time by default) stamps
    from the data.

    Args:
        data (xarray dataset): data to remove duplicates from

    Returns:
        xarray dataset: data without duplicate time stamps
    """
    # remove duplicate time stamps
    return data.drop_duplicates(dimension)


def split_data(data, dimension="time"):
    """
    Split the data into training, validation, and testing sets.

    Args:
        data (xarray dataset): data to split

    Returns:
        dict: dictionary containing the training, validation, and testing sets
    """
    # get total range for the data in the specified dimension
    total_range = data[dimension].values

    # break the range into 3 parts at ratios 70:15:15
    train_range = total_range[:int(0.7 * len(total_range))]
    val_range = total_range[int(0.7 * len(total_range)):int(0.85 * len(total_range))]
    test_range = total_range[int(0.85 * len(total_range)):]

    # split the data into training, validation, and testing sets
    training = data.sel(time=train_range)
    validation = data.sel(time=val_range)
    testing = data.sel(time=test_range)

    return {"train": training, "val": validation, "test": testing}