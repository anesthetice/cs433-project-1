from collections.abc import Generator
from collections.abc import Callable
import numpy as np
import numpy.typing as npt

# Copied from the dev branch of numpy (will be available in 2.6 according to their online ref)
type Array1D[ST: np.generic] = np.ndarray[tuple[int], np.dtype[ST]]
type Array2D[ST: np.generic] = np.ndarray[tuple[int, int], np.dtype[ST]]


def compute_lr_mse_loss(
    y: Array1D,
    tx: Array2D,
    w: Array1D,
) -> float:
    """Computes the MSE loss for linear regression."""
    e = y - tx@w
    mse = (1/2) * np.mean(np.square(e))
    return mse


def compute_lr_mse_grad(
    y: Array1D,
    tx: Array2D,
    w: Array1D,
) -> Array1D:
    """Computes the gradient at w for linear regression using MSE as the loss function."""
    N = y.shape[0]
    e = y - tx@w
    return (-1/N) * tx.T @ e


def _gradient_descent_generic(
    batch_gen: Generator[tuple[Array1D, Array2D]],
    compute_loss_func: Callable[[...], float],
    compute_grad_func: Callable[[...], Array1D],
    initial_w: Array1D,
    gamma: float,
    append_final_loss_flag: bool = True,
) -> tuple[Array1D, list[float]]:
    """
    Generic private function for gradient descent. I hate code duplication.
    But to be fair this might not be too readable either.
    """
    w: Array1D = initial_w
    losses: list[float] = []

    for (n_iter, (y, tx)) in enumerate(batch_gen):

        losses.append(compute_loss_func(y, tx, w))
        grad: Array1D = compute_grad_func(y, tx, w)

        w = w - gamma * grad

    if append_final_loss_flag:
        final_loss = compute_loss_func(y, tx, w)
        losses.append(final_loss)

    return w, losses

def mean_squared_error_gd(
    y: Array1D,
    tx: Array2D,
    initial_w: Array1D,
    max_iters: int,
    gamma: float,
    append_final_loss_flag: bool = True
):
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

    def create_batch_gen(
        y: Array1D,
        tx: Array2D,
        max_iters: int = max_iters
    ):
        n_iter = 0
        while n_iter < max_iters:
            yield y, tx

    return _gradient_descent_generic(
        batch_gen = create_batch_gen(y, tx, max_iters),
        compute_loss_func = compute_lr_mse_loss,
        compute_grad_func = compute_lr_mse_grad,
        initial_w = initial_w,
        gamma = gamma,
        append_final_loss_flag = append_final_loss_flag,
    )


def mean_squared_error_sgd(
    y: Array1D,
    tx: Array2D,
    initial_w: Array1D,
    max_iters: int,
    gamma: float,
    batch_size: int = 1,
    append_final_loss_flag: bool = True
):
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
            list containing the loss at every epoch, has a lenght of `max_iters+1` if append_final_loss_flag is set to `True`.
    """

    def create_batch_gen(
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

    return _gradient_descent_generic(
        batch_gen = create_batch_gen(y, tx, max_iters, batch_size),
        compute_loss_func = compute_lr_mse_loss,
        compute_grad_func = compute_lr_mse_grad,
        initial_w = initial_w,
        gamma = gamma,
        append_final_loss_flag = append_final_loss_flag,
    )


def least_squares(
    y: Array1D,
    tx: Array2D,
) -> Array1D:
    """Computes the least squares solution.

    Args:
        y: numpy array of shape (N,) where N is the number of samples.
        tx: numpy array of shape (N,D) where D is the number of features/dimensions.

    Returns:
        w: vector of the optimal model weights, a numpy array of shape (D,) where D is the number of features/dimensions.
        MSE: a scalar representing the mean squared error.
    """
    # We wish to solve the following: X^T X w = X^T y
    lhs = tx.T @ tx
    rhs = tx.T @ y
    w_ls = np.linalg.solve(lhs, rhs)
    mse = compute_lr_mse_loss(y, tx, w_ls)
    return w_ls, mse
