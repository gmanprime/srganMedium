import os
import xarray as xr


def get_filenames(path):
    """
    Generate a list of .nc files in the specified folder.

    This method scans the directory specified by the `self.path` attribute
    and returns a list of filenames that have a `.nc` extension.

    Args:
        path (str): path to the folder to scan

    Returns:
        list: A list of filenames with a `.nc` extension in the specified folder.
    """

    # generate a list of .nc files in the folder
    return [f for f in os.listdir(path) if f.endswith(".nc")]


def save(data, path="../data/"):
    """
    Save the data to the specified path

    Args:
        data (xarray dataset): data to save
        path (str): path to save the data
    Raises:
        ValueError: if data is not an xarray dataset
    """

    # check to see if save path exists
    # if not create the path (including subfolder's)
    if not os.path.exists(path):
        os.makedirs(path)

    # save the data if the data is an xarray dataset
    if isinstance(data, xr.Dataset):
        data.to_netcdf(path)
    else:
        raise ValueError("Data must be an xarray dataset")


def load(path, file):
    """
    Load the data from the specified file

    Args:
        path (str): path to the file
        file (str): name of the file to load

    Returns:
        xarray dataset: loaded data
    """
    # load the data into the preprocessing module
    return xr.open_dataset(path + file)
