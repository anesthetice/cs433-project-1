from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from itertools import islice

import numpy as np
import numpy.typing as npt

# Copied from the dev branch of numpy (will be available in 2.6 according to their online ref)
type Array1D[ST: np.generic] = np.ndarray[tuple[int], np.dtype[ST]]
type Array2D[ST: np.generic] = np.ndarray[tuple[int, int], np.dtype[ST]]


@dataclass(init=False)
class DataLoader:
    n: int
    y_: Array1D
    x_: Array2D
    rng: np.random.Generator
    batch_size: int

    def __init__(
        self,
        y: Array1D,
        x: Array2D,
        batch_size: int | None = None,
        rng_seed: int | None = None,
    ):
        assert y.shape[0] == x.shape[0]
        if batch_size is not None:
            assert batch_size > 0

        self.n = y.shape[0]
        self.y_ = y
        self.x_ = x
        self.rng = np.random.default_rng(rng_seed)
        self.batch_size = self.n if batch_size is None else batch_size

    @property
    def y(self) -> Array1D:
        return self.y_

    @property
    def x(self) -> Array2D:
        return self.x_

    @property
    def data(self) -> tuple[Array1D, Array2D]:
        return self.y_, self.x_

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Array1D, Array2D]:
        if self.batch_size == self.n:
            return self.data
        else:
            indices_for_batch: Array1D = self.rng.choice(
                np.arange(self.n), size=self.batch_size,
                replace=False,
            )
            return self.y_[indices_for_batch], self.x_[indices_for_batch, :]


def _gradient_descent_generic(
    dataloader: DataLoader,
    compute_loss_func: Callable[[...], float],
    compute_grad_func: Callable[[...], Array1D],
    initial_w: Array1D,
    max_iters: int,
    gamma: float,
    _debug_flag: bool = True
) -> tuple[Array1D, float]:
    """
    Generic function for gradient descent. I hate code duplication.
    But to be fair this might not be too readable either.

    Parameters
    ----------
    dataloader : DataLoader
        The poor man's alternative to a PyTorch dataloader.
    compute_loss_func: Callable[[...], float],
        The function used to compute the loss, called using `compute_loss_func(y, tx, w)`.
    compute_grad_func: Callable[[...], Array1D],
        The function used to compute the gradient, called using `compute_grad_func(y, tx, w)`.
    initial_w : Array1D
        The initial weights.
    max_iters : int
        The maximum number of iterations.
    gamma : float
        The step size parameter used.

    Returns
    -------
    Array1D
        The final weights of the model.
    float:
        The final loss of the model.
    """
    w: Array1D = initial_w

    for n_iter, (y, tx) in enumerate(islice(dataloader, max_iters)):
        if _debug_flag:
            loss = compute_loss_func(*dataloader.data, w)
            print(f"Epoch {n_iter}, loss = {loss:.3f}")
        grad: Array1D = compute_grad_func(y, tx, w)
        w = w - gamma * grad

    final_loss = compute_loss_func(*dataloader.data, w)
    return w, final_loss


def compute_lr_mse_loss(
    y: Array1D,
    tx: Array2D,
    w: Array1D,
) -> float:
    """Computes the MSE loss for linear regression."""
    e = y - tx @ w
    mse = (1 / 2) * np.mean(np.square(e))
    return mse


def compute_lr_mse_grad(
    y: Array1D,
    tx: Array2D,
    w: Array1D,
) -> Array1D:
    """Computes the gradient at w for linear regression using MSE as the loss function."""
    N = y.shape[0]
    e = y - tx @ w
    return (-1 / N) * tx.T @ e


def mean_squared_error_gd(
    y: Array1D,
    tx: Array2D,
    initial_w: Array1D,
    max_iters: int,
    gamma: float,
) -> tuple[Array1D, float]:
    """The Gradient Descent (GD) algorithm for linear regression using MSE as the cost function.

    Parameters
    ----------
    y : Array1D
        The observed output samples, an array of shape (N,) where N is the number of samples.
    tx : Array2D
        The observed input samples, a matrix of shape (N, D) where D is the number of dimensions/features.
    initial_w : Array1D
        The initial weights.
    max_iters : int
        The maximum number of iterations.
    gamma : float
        The step size parameter used.

    Returns
    -------
    Array1D
        The final weights of the model.
    float:
        The final loss of the model.
    """

    return _gradient_descent_generic(
        dataloader=DataLoader(y, tx),
        compute_loss_func=compute_lr_mse_loss,
        compute_grad_func=compute_lr_mse_grad,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
    )


def mean_squared_error_sgd(
    y: Array1D,
    tx: Array2D,
    initial_w: Array1D,
    max_iters: int,
    gamma: float,
    batch_size: int = 1,
) -> tuple[Array1D, float]:
    """The Stochastic Gradient Descent (SGD) algorithm for linear regression using MSE as the cost function.

    Parameters
    ----------
    y : Array1D
        The observed output samples, an array of shape (N,) where N is the number of samples.
    tx : Array2D
        The observed input samples, a matrix of shape (N, D) where D is the number of dimensions/features.
    initial_w : Array1D
        The initial weights.
    max_iters : int
        The maximum number of iterations.
    gamma : float
        The step size parameter used.

    Returns
    -------
    Array1D
        The final weights of the model.
    float:
        The final loss of the model.
    """

    return _gradient_descent_generic(
        dataloader=DataLoader(y, tx, batch_size),
        compute_loss_func=compute_lr_mse_loss,
        compute_grad_func=compute_lr_mse_grad,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
    )


def least_squares(
    y: Array1D,
    tx: Array2D,
) -> tuple[Array1D, float]:
    """Computes the least squares solution.

    Parameters
    ----------
    y : Array1D
        The observed output samples, an array of shape (N,) where N is the number of samples.
    tx : Array2D
        The observed input samples, a matrix of shape (N, D) where D is the number of dimensions/features.

    Returns
    -------
    Array1D
        The computed optimal weights of the model, an array of shape (D,).
    float:
        The (MSE) loss of the model.
    """
    # We wish to solve the following: (X^T X) w = X^T y
    lhs = tx.T @ tx
    rhs = tx.T @ y
    w_ls = np.linalg.solve(lhs, rhs)
    mse = compute_lr_mse_loss(y, tx, w_ls)
    return w_ls, mse


def ridge_regression(
    y: Array1D,
    tx: Array2D,
    lambda_: float,
) -> tuple[Array1D, float]:
    """Computes the least squares solution with regularization.

    Parameters
    ----------
    y : Array1D
        The observed output samples, an array of shape (N,) where N is the number of samples.
    tx : Array2D
        The observed input samples, a matrix of shape (N, D) where D is the number of dimensions/features.
    lambda_ : float
        The λ term used in regularization.

    Returns
    -------
    Array1D
        The computed optimal weights of the model, an array of shape (D,).
    float:
        The loss of the model.
    """
    # We wish to solve the following: (X^T X λ'I) w = X^T y
    # Where λ' = 2Nλ

    N, D = tx.shape
    lhs = tx.T @ tx + (2 * N * lambda_ * np.identity(D))
    rhs = tx.T @ y
    w_ls_ridge = np.linalg.solve(lhs, rhs)
    mse = compute_lr_mse_loss(y, tx, w_ls_ridge)
    return w_ls_ridge, mse


def logistic_func(eta: npt.ArrayLike) -> np.ndarray:
    """The logistic function, which is typically denoted by σ(η)."""
    return 1.0 / (1.0 + np.exp(-eta))


def compute_logreg_loss(
    y: Array1D,
    tx: Array2D,
    w: Array1D,
) -> float:
    """Computes the loss for logistic regression for labels in {-1, +1}."""
    return np.mean(np.logaddexp(0.0, -(y * (tx @ w))))
    # Old approach, works for {0, +1}, but incorrect for {-1, +1}
    # From the expression (1/N) * sum_{n=1}^{N} -y_n * x_n^T w + ln(1 + exp(x_n^T w))
    # return np.mean(-y * log_odds + np.logaddexp(0, log_odds))


def compute_logreg_grad(
    y: Array1D,
    tx: Array2D,
    w: Array1D,
) -> Array1D:
    """Computes the gradient used in logistic regression for labels in {-1, +1}."""
    N = y.shape[0]
    return -(tx.T @ (y * logistic_func(-y * (tx @ w)))) / N
    # Old approach, works for {0, +1}, but incorrect for {-1, +1}
    # return (1 / N) * tx.T @ (logistic_func(tx @ w) - y)


def logistic_regression(
    y: Array1D,
    tx: Array2D,
    initial_w: Array1D,
    max_iters: int,
    gamma: float,
) -> tuple[Array1D, float]:
    """
    Finds the model parameters for logistic regression using gradient descent.

    Parameters
    ----------
    y : Array1D
        Our observed binary outputs, an array of shape (N,) where N is the number of samples,
        the values contained within should be either zero or one.
    tx : Array2D
        Our observed inputs, a matrix of shape (N, D) where D is the number of features/dimensions.
    initial_w : Array1D
        The initial weights.
    max_iters : int
        The maximum number of iterations.
    gamma : float
        The step size parameter used.

    Returns
    -------
    Array1D
        The final weights of the model.
    float:
        The final loss of the model.
    """

    return _gradient_descent_generic(
        dataloader=DataLoader(y, tx),
        compute_loss_func=compute_logreg_loss,
        compute_grad_func=compute_logreg_grad,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
    )


def compute_logreg_grad_reg(
    y: Array1D, tx: Array2D, w: Array1D, lambda_: float
) -> Array1D:
    """Computes the gradient used in logistic regression."""
    return  compute_logreg_grad(y, tx, w) + 2 * lambda_ * w


def compute_logreg_loss_reg(
    y: Array1D,
    tx: Array2D,
    w: Array1D,
    lambda_: float,
) -> float:
    return compute_logreg_loss(y, tx, w) + lambda_ * np.sum(w**2)


def reg_logistic_regression(
    y: Array1D,
    tx: Array2D,
    lambda_: float,
    initial_w: Array1D,
    max_iters: int,
    gamma: float,
) -> tuple[Array1D, float]:
    """
    Finds the model parameters for logistic regression with regularization using gradient descent.

    Parameters
    ----------
    y : Array1D
        Our observed binary outputs, an array of shape (N,) where N is the number of samples,
        the values contained within should be either zero or one.
    tx : Array2D
        Our observed inputs, a matrix of shape (N, D) where D is the number of features/dimensions.
    lambda_ : float
        The λ term used for regularization.
    initial_w : Array1D
        The initial weights.
    max_iters : int
        The maximum number of iterations.
    gamma : float
        The step size parameter used.

    Returns
    -------
    Array1D
        The final weights of the model.
    float:
        The final loss of the model.
    """

    return _gradient_descent_generic(
        dataloader=DataLoader(y, tx),
        compute_loss_func=partial(compute_logreg_loss_reg, lambda_=lambda_),
        compute_grad_func=partial(compute_logreg_grad_reg, lambda_=lambda_),
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
    )
