from collections.abc import Generator
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from functools import partial

# Copied from the dev branch of numpy (will be available in 2.6 according to their online ref)
type Array1D[ST: np.generic] = np.ndarray[tuple[int], np.dtype[ST]]
type Array2D[ST: np.generic] = np.ndarray[tuple[int, int], np.dtype[ST]]


def _gradient_descent_generic(
    dataloader: Generator[tuple[Array1D, Array2D]],
    compute_loss_func: Callable[[...], float],
    compute_grad_func: Callable[[...], Array1D],
    initial_w: Array1D,
    gamma: float,
    append_final_loss_flag: bool = True,
) -> tuple[Array1D, list[float]]:
    """
    Generic private function for gradient descent. I hate code duplication.
    But to be fair this might not be too readable either.

    Parameters
    ----------
    dataloader : Generator[tuple[Array1D, Array2D]]
        The poor man's alternative to a pytorch dataloader.
    compute_loss_func: Callable[[...], float],
        The function used to compute the loss, called using `compute_loss_func(y, tx, w)`.
    compute_grad_func: Callable[[...], Array1D],
        The function used to compute the gradient, called using `compute_grad_func(y, tx, w)`.
    initial_w : Array1D
        The initial weights.
    gamma : float
        The step size parameter used.
    append_final_loss_flag : bool
        Decides if the loss is computed at the end and appended to the losses,
        not that the losses are computed at the start of each epoch and not at the end.

    Returns
    -------
    Array1D
        The final weights of the model.
    list[float]:
        The losses at each epoch, index with -1 to get the (semi-)final loss.
    """
    w: Array1D = initial_w
    losses: list[float] = []

    for n_iter, (y, tx) in enumerate(dataloader):
        losses.append(compute_loss_func(y, tx, w))
        grad: Array1D = compute_grad_func(y, tx, w)

        w = w - gamma * grad

    if append_final_loss_flag:
        final_loss = compute_loss_func(y, tx, w)
        losses.append(final_loss)

    return w, losses


def create_gd_dataloader(y: Array1D, tx: Array2D, max_iters: int):
    n_iter = 0
    while n_iter < max_iters:
        yield y, tx


def create_sgd_dataloader(
    y: Array1D,
    tx: Array2D,
    max_iters: int,
    batch_size: int,
):
    N = y.shape[0]
    indices: Array1D = np.arange(N)
    rng = np.random.default_rng()

    n_iter = 0
    while n_iter < max_iters:
        indices_for_batch: Array1D = rng.choice(indices, size=batch_size)
        yield y[indices_for_batch], tx[indices_for_batch, :]


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
    append_final_loss_flag: bool = True,
) -> tuple[Array1D, list[float]]:
    """The Gradient Descent (GD) algorithm for linear regression using MSE as the cost function.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the vector containing the final weights of the model
        losses:
            list containing the loss at every epoch, has a lenght of `max_iters+1` if append_final_loss_flag is set to `True`.
    """

    return _gradient_descent_generic(
        dataloader=create_gd_dataloader(y, tx, max_iters),
        compute_loss_func=compute_lr_mse_loss,
        compute_grad_func=compute_lr_mse_grad,
        initial_w=initial_w,
        gamma=gamma,
        append_final_loss_flag=append_final_loss_flag,
    )


def mean_squared_error_sgd(
    y: Array1D,
    tx: Array2D,
    initial_w: Array1D,
    max_iters: int,
    gamma: float,
    batch_size: int = 1,
    append_final_loss_flag: bool = True,
) -> tuple[Array1D, list[float]]:
    """The Stochastic Gradient Descent (SGD) algorithm for linear regression using MSE as the cost function.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the vector containing the final weights of the model
        losses:
            list containing the loss at every epoch, has a lenght of `max_iters+1` if append_final_loss_flag is set to `True`
    """

    return _gradient_descent_generic(
        dataloader=create_sgd_dataloader(y, tx, max_iters, batch_size),
        compute_loss_func=compute_lr_mse_loss,
        compute_grad_func=compute_lr_mse_grad,
        initial_w=initial_w,
        gamma=gamma,
        append_final_loss_flag=append_final_loss_flag,
    )


def least_squares(
    y: Array1D,
    tx: Array2D,
) -> tuple[Array1D, float]:
    """Computes the least squares solution.

    Args:
        y: numpy array of shape (N,) where N is the number of samples
        tx: numpy array of shape (N,D) where D is the number of features/dimensions

    Returns:
        w: vector of the optimal model weights, a numpy array of shape (D,) where D is the number of features/dimensions
        MSE: a scalar representing the mean squared error
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
    """Computes the least squares solution.

    Args:
        y: numpy array of shape (N,) where N is the number of samples
        tx: numpy array of shape (N,D) where D is the number of features/dimensions
        lambda_: the λ term (scalar) used in ridge regression

    Returns:
        w: vector of the optimal model weights, a numpy array of shape (D,) where D is the number of features/dimensions
        MSE: a scalar representing the mean squared error
    """
    # We wish to solve the following: (X^T X λ'I) w = X^T y
    # Where λ' = 2Nλ
    N = y.shape[0]
    lhs = tx.T @ tx @ (2 * N * lambda_ * np.identity(N))
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
    """Computes the loss for logistic regression."""
    log_odds = tx @ w
    # From the expression (1/N) * sum_{n=1}^{N} -y_n * x_n^T w + ln(1 + exp(x_n^T w))
    return np.mean(-y * log_odds + np.logaddexp(0, log_odds))


def compute_logreg_grad(
    y: Array1D,
    tx: Array2D,
    w: Array1D,
) -> Array1D:
    """Computes the gradient used in logistic regression."""
    N = y.shape[0]
    return (1 / N) * tx.T @ (logistic_func(tx @ w) - y)


def logistic_regression(
    y: Array1D,
    tx: Array2D,
    initial_w: Array1D,
    max_iters: int,
    gamma: float,
    append_final_loss_flag: bool = True,
) -> tuple[Array1D, list[float]]:
    return _gradient_descent_generic(
        dataloader=create_gd_dataloader(y, tx, max_iters),
        compute_loss_func=compute_logreg_loss,
        compute_grad_func=compute_logreg_grad,
        initial_w=initial_w,
        gamma=gamma,
        append_final_loss_flag=append_final_loss_flag,
    )


def compute_logreg_grad_reg(
    y: Array1D, tx: Array2D, w: Array1D, lambda_: float
) -> Array1D:
    """Computes the gradient used in logistic regression."""
    N = y.shape[0]
    return (1 / N) * tx.T @ (logistic_func(tx @ w) - y) + 2 * lambda_ * w


def reg_logistic_regression(
    y: Array1D,
    tx: Array2D,
    lambda_: float,
    initial_w: Array1D,
    max_iters: int,
    gamma: float,
    append_final_loss_flag: bool = True,
) -> tuple[Array1D, list[float]]:
    return _gradient_descent_generic(
        dataloader=create_gd_dataloader(y, tx, max_iters),
        compute_loss_func=compute_logreg_loss,
        compute_grad_func=partial(compute_logreg_grad_reg, lambda_=lambda_),
        initial_w=initial_w,
        gamma=gamma,
        append_final_loss_flag=append_final_loss_flag,
    )
