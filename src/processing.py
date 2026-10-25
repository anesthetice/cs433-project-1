from implementations import Array1D, Array2D
from helpers import load_data
from consts import IDX_TO_COLNAME, COLNAME_TO_IDX, INTERESTING_FEATURES

import numpy as np
import numpy.typing as npt

from os import PathLike
from typing import Literal

def process_feature(
    col: Array1D,
    colname: str,
    feature_type: Literal["cat", "ordcat", "numerical"],
    unknown_values: list[float] = [],
    replace_nan_with: int | None = None,
    replace_vals: list[tuple[float, float]] | None = None
):
    N = col.shape[0]
    print(f"Working on the {feature_type} feature `{colname}`")

    if replace_vals is not None:
        for (v1, v2) in replace_vals:
            col[col == v1] = v2

    # np.isin() with NaN just doesn't work
    nan_mask = ~np.isfinite(col)

    match feature_type:
        case "cat":
            uniques: Array1D = np.unique(col)
            print(f" - pre-uniques: {uniques}")
            uniques = uniques[
                np.isfinite(uniques) & ~np.isin(uniques, unknown_values, assume_unique=True)
            ] # ~ is the logical not operator
            print(f" - uniques: {uniques}")

            output: Array2D = np.zeros(shape=(N, len(uniques) + 1), dtype=np.float64)

            cidx = 0
            for val in uniques:
                output[:, cidx] = (col == val)
                cidx += 1

            output[:, cidx] = np.isin(col, unknown_values) | nan_mask

            return output

        case "ordcat" | "numerical":
            if len(unknown_values) == 0 and nan_mask.sum() == 0:
                    return col.astype(np.float64).reshape((N, 1))

            output: Array2D = np.zeros(shape=(N, 2), dtype=np.float64)
            output[:, 0] = col
            output[nan_mask, 0] = replace_nan_with
            output[:, 1] = np.isin(col, unknown_values) | nan_mask

            return output

        case _:
            raise ValueError


def standardize(x: Array2D) -> Array2D:
    x = x - np.mean(x, axis=0)
    x = x / np.std(x, axis=0)
    return x

def load_and_preprocess(
    data_path: PathLike = "./data/", sub_sample: bool = False
) -> tuple[Array2D, Array1D, Array2D]:
    x_train, x_test, y_train, _, _ = load_data(data_path)

    N_train = x_train.shape[0]
    N_test = x_test.shape[0]

    x_full = np.vstack((x_train, x_test))
    x_full = standardize(x_full)
    assert(x_full.shape[0] == N_train + N_test)

    parts: list[np.ndarray] = []
    for entry in INTERESTING_FEATURES:
        colname: str = entry["colname"]
        colidx: int = COLNAME_TO_IDX[colname]

        processed = process_feature(
            col=x_full[:, colidx],
            colname=colname,
            feature_type=entry["type"],
            unknown_values=entry["uvs"],
            replace_nan_with=entry["rnw"] if "rnw" in entry.keys() else None,
            replace_vals=entry["replace"] if "replace" in entry.keys() else None,
        )

        parts.append(processed)

    x_full_out = np.hstack(parts)
    assert(x_full_out.shape[0] == N_train + N_test)

    return x_full_out[:N_train, :], y_train, x_full_out[N_train:, :]
