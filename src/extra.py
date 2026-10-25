import numpy as np

from implementations import Array1D, Array2D


def train_test_split(
    y: Array1D,
    x: Array2D,
    train_to_test_ratio: float = 0.8,
    rng_seed: int | None = None,
) -> tuple[tuple[Array1D, Array2D], tuple[Array1D, Array2D]]:
    assert train_to_test_ratio >= 0.0 and train_to_test_ratio <= 1.0
    assert x.shape[0] == y.shape[0]
    N = y.shape[0]

    indices: Array1D = np.arange(N)
    np.random.default_rng(rng_seed).shuffle(indices)

    cutoff_idx = round(N * train_to_test_ratio)
    train_indices: Array1D = indices[:cutoff_idx]
    test_indices: Array1D = indices[cutoff_idx:]

    return (y[train_indices], x[train_indices]), (y[test_indices], x[test_indices])


def see_statistics(
    y_true: Array1D,
    y_pred: Array1D,
):
    TP = np.sum((y_true == +1) & (y_pred == +1))
    TN = np.sum((y_true == -1) & (y_pred == -1))

    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))

    print(f"""
    TP: {TP}
    TN: {TN}

    FP: {FP}
    FN: {FN}
    """)
