from os import PathLike
from typing import Literal

import numpy as np

from consts import COLNAME_TO_IDX, INTERESTING_FEATURES
from helpers import load_data
from implementations import Array1D, Array2D


def process_feature(
    col: Array1D,
    colname: str,
    feature_type: Literal["cat", "ordcat", "numerical"],
    unknown_values: list[float] = [],
    replace_vals: list[tuple[float, float]] | None = None,
) -> Array2D:
    N = col.shape[0]
    print(f"Working on the {feature_type} feature `{colname}`")

    if replace_vals is not None:
        for v1, v2 in replace_vals:
            col[col == v1] = v2

    # np.isin() with NaN just doesn't work
    nan_mask = ~np.isfinite(col)

    match feature_type:
        case "cat":
            uniques: Array1D = np.unique(col)
            print(f" * pre-uniques: {uniques}")
            uniques = uniques[
                np.isfinite(uniques)
                & ~np.isin(uniques, unknown_values, assume_unique=True)
            ]  # ~ is the logical not operator
            print(f" * uniques: {uniques}")

            output: Array2D = np.zeros(shape=(N, len(uniques) + 1), dtype=np.float64)

            cidx = 0
            for val in uniques:
                output[:, cidx] = col == val
                cidx += 1

            output[:, cidx] = np.isin(col, unknown_values) | nan_mask

            return output

        case "ordcat":
            if len(unknown_values) == 0 and nan_mask.sum() == 0:
                return col.astype(np.float64).reshape((N, 1))

            valid_mask: Array1D = (~nan_mask) & (~np.isin(col, unknown_values))

            median: float = np.median(col[valid_mask])

            print(f" * median (used as imputer): {median:.2f}")

            col[~valid_mask] = median

            output: Array2D = np.zeros(shape=(N, 2), dtype=np.float64)
            output[:, 0] = col
            output[:, 1] = ~valid_mask
            return output

        case "numerical":
            if len(unknown_values) == 0 and nan_mask.sum() == 0:
                return col.astype(np.float64).reshape((N, 1))

            valid_mask: Array1D = (~nan_mask) & (~np.isin(col, unknown_values))

            median: float = np.median(col[valid_mask])
            mean: float = np.mean(col[valid_mask])
            std: float = np.std(col[valid_mask])

            print(f" * median (used as imputer): {median:.2f}")
            print(f" * μ: {mean:.2f}, σ: {std:.2f}, max: {np.max(col[valid_mask]):.2f}")

            col[~valid_mask] = median

            # maybe use old values here?
            col = col - np.mean(col)
            col = col / np.std(col)

            output: Array2D = np.zeros(shape=(N, 2), dtype=np.float64)
            output[:, 0] = col
            output[:, 1] = ~valid_mask
            return output

        case _:
            raise ValueError


def load_and_preprocess(
    target_features: list[dict] = INTERESTING_FEATURES,
    data_path: PathLike = "./data/",
    sub_sample: bool = False,
) -> tuple[Array2D, Array1D, Array2D, list[str]]:
    x_train, x_test, y_train, _, _ = load_data(data_path, sub_sample)

    N_train = x_train.shape[0]
    N_test = x_test.shape[0]

    x_full = np.vstack((x_train, x_test))
    assert x_full.shape[0] == N_train + N_test

    parts: list[np.ndarray] = []
    colidx_to_colname: list[str] = []
    for entry in target_features:
        colname: str = entry["colname"]
        colidx: int = COLNAME_TO_IDX[colname]

        processed = process_feature(
            col=x_full[:, colidx],
            colname=colname,
            feature_type=entry["type"],
            unknown_values=entry["uvs"],
            replace_vals=entry["replace"] if "replace" in entry else None,
        )

        parts.append(processed)
        colidx_to_colname.extend([colname] * processed.shape[1])

    x_full_out = np.hstack(parts)
    assert x_full_out.shape[0] == N_train + N_test

    return x_full_out[:N_train, :], y_train, x_full_out[N_train:, :], colidx_to_colname
