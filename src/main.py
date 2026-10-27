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

    (w_best, tr_best, stats_best, iter_best), train_losses, test_losses = (
        extra.reg_logistic_regression_f1(
            y_train,
            x_train,
            y_test,
            x_test,
            lambda_value=1e-5,
            initial_w=np.zeros(x_train.shape[1]),
            gamma=0.6,
            n_iters=300,
            rng_seed=RNG_SEED,
        )
    )

    print(
        f"\nBest iter n°{iter_best}: F1 = {stats_best['F1']} (threshold = {tr_best})"
    )


if __name__ == "__main__":
    main()
