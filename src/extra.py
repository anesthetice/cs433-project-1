from consts import INTERESTING_FEATURES
from preprocessing import load_and_preprocess
from itertools import islice
import numpy as np
from os import PathLike
import math
from copy import deepcopy
from implementations import (
    Array1D,
    Array2D,
    compute_logreg_grad_reg,
    compute_logreg_loss_reg,
    DataLoader,
)


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


def standardize_and_add_bias(
    x_train: Array2D,
    x_valid: Array2D,
    x_test: Array2D,
) -> tuple[Array2D, Array2D, Array2D]:
    """Standardizes every column using only the training set's stats, then adds a column of ones (bias)."""
    assert x_train.shape[1] == x_valid.shape[1] == x_test.shape[1]

    mean: Array1D = x_train.mean(axis=0)
    std: Array1D = x_train.std(axis=0)
    std[std == 0] = 1.0

    outputs = []
    for x in (x_train, x_valid, x_test):
        x = (x - mean) / std
        outputs.append(np.hstack((np.ones((x.shape[0], 1)), x)))

    return outputs[0], outputs[1], outputs[2]


def get_pseudo_train_test_sets(
    target_features: list[dict] = INTERESTING_FEATURES,
    data_path: PathLike = "./data/",
    sub_sample: bool = False,
    train_to_test_ratio: float = 0.8,
    rng_seed: int | None = None,
) -> tuple[tuple[Array1D, Array2D], tuple[Array1D, Array2D], list[str]]:

    x_train_full, y_train_full, _, colidx_to_colname = load_and_preprocess(
        target_features, data_path, sub_sample
    )

    return *train_test_split(
        y_train_full, x_train_full, train_to_test_ratio, rng_seed
    ), colidx_to_colname


def logreg_predict(x: Array2D, w: Array1D, threshold: float = 0.5) -> Array1D:
    N, _ = x.shape
    y_pred = -np.ones(N)
    y_prob = 1 / (1 + np.exp(-x @ w))
    y_pred[y_prob > threshold] = +1
    return y_pred


def compute_statistics(
    y_true: Array1D,
    y_pred: Array1D,
) -> dict[str, float]:
    # Actual total positive
    P = np.sum(y_true == +1)
    # Actual total negative
    N = np.sum(y_true == -1)

    # Predicted total positive
    PP = np.sum(y_pred == +1)
    # Predicted total negative
    PN = np.sum(y_pred == -1)

    TP = np.sum((y_true == +1) & (y_pred == +1))
    FN = np.sum((y_true == +1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == +1))
    TN = np.sum((y_true == -1) & (y_pred == -1))

    # Also known as the true positive ratio (TPR), it's the probability of a positive test result, conditioned on the individual truly being positive.
    sensitivity = TP / P  # P = TP + FN

    # Also known as the true negative ratio (TNR), or recall, it's the probability of a negative test result, conditioned on the individual truly being negative.
    specificity = TN / N  # N = FP + TN

    # Also known as the positive predictive value (PPV), it's the probability of the individual truly being positive, conditioned on a positive test result.
    precision = TP / PP if PP > 0 else 0.0  # PP = TP + FP

    # Harmonic mean of precision and recall, this is the metric used to rank submissions.
    f1 = 2 * TP / (2 * TP + FP + FN) if TP > 0 else 0.0

    return {
        "P": P,
        "N": N,
        "PP": PP,
        "PN": PN,
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "TN": TN,
        "sensitivity": sensitivity,
        "TPR": sensitivity,
        "specificity": specificity,
        "TNR": specificity,
        "precision": precision,
        "PPV": precision,
        "F1": f1,
    }


def find_best_threshold(
    y_true: Array1D,
    x: Array2D,
    w: Array1D,
    thresholds: Array1D = np.linspace(0, 1, 201),
) -> tuple[float, dict[str, float]]:
    """Finds the threshold that maximizes the F1 score, returns it with its stats."""
    y_prob = 1 / (1 + np.exp(-x @ w))

    tr_best = 0.5
    stats_best: dict[str, float] = {"F1": -1.0}

    for tr in thresholds:
        y_pred = -np.ones_like(y_true)
        y_pred[y_prob > tr] = +1
        stats = compute_statistics(y_true=y_true, y_pred=y_pred)

        if stats["F1"] > stats_best["F1"]:
            tr_best = tr
            stats_best = stats

    return tr_best, stats_best


def reg_logistic_regression_2(
    y_train: Array1D,
    x_train: Array2D,
    y_test: Array1D,
    x_test: Array2D,
    lambda_value: float,
    initial_w: Array1D | None = None,
    gamma: float = 0.6,
    n_iters: int = 100,
    batch_size: int | None = 8192,
    rng_seed: int | None = None,
) -> tuple[Array1D, list[float], list[float]]:
    """
    Finds the model parameters for logistic regression with regularization using gradient descent.

    Parameters
    ----------
    y_train : Array1D
        Our observed binary outputs, an array of shape (N,) where N is the number of samples,
        the values contained within should be either zero or one.
    x_train : Array2D
        Our observed inputs, a matrix of shape (N, D) where D is the number of features/dimensions.
    y_test : Array1D
        The test set used to compute statistics.
    x_test : Array2D
        The test set used to compute statistics.
    lambda_value : float
        The λ term used for regularization.
    initial_w : Array1D | None
        The initial weights used, by default these will be generated by rng.
    gamma : float
        The step size parameter used.
    n_iters : int
        The number of iterations performed.
    batch_size : int
        The batch sized used for SGD. If set to none, equivalent to GD (everything is used)
    rng_seed : int | None
        The seed that is fed downstream to `np.random.default_rng()`.

    Returns
    -------
    tuple[Array1D, float, int]
        The final weights of the best model, its associated testing loss, and iteration number.
    list[float]
        The training loss at each step.
    list[float]
        The testing loss at each step.
    """

    N, D = x_train.shape
    rng = np.random.default_rng(rng_seed)
    train_losses = []
    test_losses = []

    w: Array1D = initial_w if initial_w is not None else rng.standard_normal(D)
    dataloader = DataLoader(y_train, x_train, batch_size, rng_seed)

    w_best: Array1D = w.copy()
    iter_best = 0
    test_loss_best: float = +math.inf

    def gamma_with_decay(curr_iter: int) -> float:
        return (gamma - 0.15 * gamma) * math.exp(
            -5 * curr_iter / n_iters
        ) + 0.15 * gamma

    for n_iter, (y, x) in enumerate(islice(dataloader, n_iters + 1)):
        train_loss = compute_logreg_loss_reg(*dataloader.data, w, lambda_value)
        train_losses.append(train_loss)

        test_loss = compute_logreg_loss_reg(y_test, x_test, w, lambda_value)
        test_losses.append(test_loss)

        # fmt: off
        print(f"Iter n°{n_iter:03}: γ = {gamma_with_decay(n_iter):.2f}")
        print(f"├─┄ training loss = {train_loss:.3f}")
        print(f"└─┄ testing loss = {test_loss}\n")
        # fmt: on

        if test_loss < test_loss_best:
            w_best = w.copy()
            iter_best = n_iter
            test_loss_best = test_loss

        grad: Array1D = compute_logreg_grad_reg(y, x, w, lambda_value)
        w = w - gamma_with_decay(n_iter) * grad

    return (w_best, test_loss_best, iter_best), train_losses, test_losses


def reg_logistic_regression_f1(
    y_train: Array1D,
    x_train: Array2D,
    y_test: Array1D,
    x_test: Array2D,
    lambda_value: float,
    initial_w: Array1D | None = None,
    gamma: float = 0.6,
    n_iters: int = 100,
    batch_size: int | None = 8192,
    rng_seed: int | None = None,
) -> tuple[tuple[Array1D, float, dict[str, float], int], list[float], list[float]]:
    """
    Same as `reg_logistic_regression_2`, but the best model is the one with the highest F1 score
    on the test set, using the best threshold for each iteration.

    Parameters
    ----------
    See `reg_logistic_regression_2`.

    Returns
    -------
    tuple[Array1D, float, dict[str, float], int]
        The final weights of the best model, its threshold, its stats, and iteration number.
    list[float]
        The training loss at each step.
    list[float]
        The testing loss at each step.
    """

    _, D = x_train.shape
    rng = np.random.default_rng(rng_seed)
    train_losses = []
    test_losses = []

    w: Array1D = initial_w if initial_w is not None else rng.standard_normal(D)
    dataloader = DataLoader(y_train, x_train, batch_size, rng_seed)

    w_best: Array1D = w.copy()
    tr_best = 0.5
    stats_best: dict[str, float] = {"F1": -1.0}
    iter_best = 0

    def gamma_with_decay(curr_iter: int) -> float:
        return (gamma - 0.15 * gamma) * math.exp(
            -5 * curr_iter / n_iters
        ) + 0.15 * gamma

    for n_iter, (y, x) in enumerate(islice(dataloader, n_iters + 1)):
        train_loss = compute_logreg_loss_reg(*dataloader.data, w, lambda_value)
        train_losses.append(train_loss)

        test_loss = compute_logreg_loss_reg(y_test, x_test, w, lambda_value)
        test_losses.append(test_loss)

        tr, stats = find_best_threshold(y_test, x_test, w)

        print(f"Iter n°{n_iter:03}: γ = {gamma_with_decay(n_iter):.2f}, test loss = {test_loss:.3f}, valid loss = {test_loss:.3f}, F1 = {stats['F1']:.3f} (threshold = {tr:.3f})")

        if stats["F1"] > stats_best["F1"]:
            w_best = w.copy()
            tr_best = tr
            stats_best = stats
            iter_best = n_iter

        grad: Array1D = compute_logreg_grad_reg(y, x, w, lambda_value)
        w = w - gamma_with_decay(n_iter) * grad

    return (w_best, tr_best, stats_best, iter_best), train_losses, test_losses


def predict_knn(
    y_train: Array1D,
    x_train: Array2D,
    x_test: Array2D,
    k: int = 5,
) -> Array1D:
    """Very slow..."""
    y_pred = np.empty(x_test.shape[0])

    for i in range(x_test.shape[0]):
        dists = np.sum(np.square(x_train - x_test[i, :]), axis=1)
        assert(dists.shape == (x_train.shape[0],))
        indices = np.argpartition(dists, k)[:k]

        y_pred[i] = +1 if y_train[indices].mean() >= 0.5 else -1

    return y_pred

_ = """
    def balanced_dl(dl: DataLoader):
        pos_indices = np.argwhere(dl.y == +1)
        neg_indices = np.argwhere(dl.y == -1)

        n_pos_to_pick = min(round(dl.batch_size * 0.5), len(pos_indices))
        n_neg_to_pick = dl.batch_size - n_pos_to_pick

        while True:

            final_pos_indices = dl.rng.choice(
                pos_indices,
                size=n_pos_to_pick,
                replace=False
            )

            final_neg_indices = dl.rng.choice(
                neg_indices,
                size=n_neg_to_pick,
                replace=False
            )

            final_indices = np.concatenate((final_pos_indices, final_neg_indices)).reshape(-1,)
            yield dl.y[final_indices], dl.x[final_indices, :]
    """
