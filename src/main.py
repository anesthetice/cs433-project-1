from extra import get_pseudo_train_test_sets
import numpy as np

import extra
from implementations import Array1D, Array2D


def main():
    RNG_SEED = 10

    (y_train, x_train), (y_test, x_test), colidx_to_colname = (
        get_pseudo_train_test_sets(
            train_to_test_ratio=0.8,
            rng_seed=RNG_SEED,
        )
    )

    (w_best, loss_best, iter_best), train_losses, test_losses = (
        extra.reg_logistic_regression_2(
            y_train,
            x_train,
            y_test,
            x_test,
            lambda_value=0.1,
            initial_w=None,
            gamma=0.3,
            n_iters=300,
            rng_seed=RNG_SEED,
        )
    )


if __name__ == "__main__":
    main()
