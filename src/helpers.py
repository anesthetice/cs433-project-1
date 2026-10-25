"""Some helper functions for project 1."""

import csv
import pathlib
from compression import zstd  # requires python 3.14
from io import TextIOWrapper
from os import PathLike

import numpy as np


def load_data(
    data_path: PathLike = "./data/", sub_sample: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    It takes like a minute to run `load_csv_data`, use this function instead.
    Will be slow the first time you run it, as the data has to be loaded from the csv at least once.

    Args:
        data_path (str, optional): The path to the data folder. Defaults to "./data/".
        sub_sample (bool, optional): If True the data will be subsempled. Defaults to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    data_path: pathlib.Path = pathlib.Path(data_path)

    if (filepath := data_path.joinpath("all.npz")).is_file():
        with np.load(filepath, allow_pickle=False) as data:
            train_ids = data["train_ids"]
            x_train = data["x_train"]
            y_train = data["y_train"]

            test_ids = data["test_ids"]
            x_test = data["x_test"]

            if sub_sample:
                y_train = y_train[::50]
                x_train = x_train[::50]
                train_ids = train_ids[::50]

            return x_train, x_test, y_train, train_ids, test_ids
    else:
        print("Must generate `all.npz` from scratch, will take longer than usual")

        print("Loading data from the csv files...")
        x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path, False)

        print("Saving data to `all.npz`...")
        np.savez_compressed(
            data_path.joinpath("all.npz"),
            train_ids=train_ids,
            x_train=x_train,
            y_train=y_train,
            test_ids=test_ids,
            x_test=x_test,
            allow_pickle=False,
        )

        if sub_sample:
            y_train = y_train[::50]
            x_train = x_train[::50]
            train_ids = train_ids[::50]

        return x_train, x_test, y_train, train_ids, test_ids


def load_csv_data(
    data_path: PathLike = "./data/", sub_sample: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Use `load_data` instead.

    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str, optional): The path to the data folder. Defaults to "./data/".
        sub_sample (bool, optional): If True the data will be subsempled. Defaults to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    data_path: pathlib.Path = pathlib.Path(data_path)

    def get_file_object(filename: str) -> TextIOWrapper:
        filename = filename.removesuffix(".zst")

        if (filepath := data_path.joinpath(filename + ".zst")).is_file():
            return zstd.open(filepath, mode="rt")
        elif (filepath := data_path.joinpath(filename)).is_file():
            return open(filepath, mode="rt")
        else:
            raise FileNotFoundError(
                f"Could not find `{filename + '.zst'}` or `{filename}` in `{data_path}`"
            )

    y_train = np.genfromtxt(
        get_file_object("y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )

    x_train = np.genfromtxt(
        get_file_object("x_train.csv"), delimiter=",", skip_header=1
    )

    x_test = np.genfromtxt(get_file_object("x_test.csv"), delimiter=",", skip_header=1)

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
