import numpy as np

import extra
from implementations import Array1D, Array2D
from processing import load_and_preprocess


def get_pseudo_train_test_sets(
    train_to_test_ratio: float = 0.8,
    rng_seed: int | None = None,
) -> tuple[tuple[Array1D, Array2D], tuple[Array1D, Array2D]]:
    x_train_full, y_train_full, _ = load_and_preprocess()

    return extra.train_test_split(
        y_train_full, x_train_full, train_to_test_ratio, rng_seed
    )


def main():
    RNG_SEED = 10

    (y_train, x_train), (y_test, x_test) = get_pseudo_train_test_sets(
        train_to_test_ratio=0.8,
        rng_seed=RNG_SEED,
    )

    (w_best, stats_best, iter_best), train_losses, test_losses = (
        extra.reg_logistic_regression_2(
            y_train,
            x_train,
            y_test,
            x_test,
            lambda_value=0.1,
            initial_w=None,
            gamma=0.3,
            max_iters=300,
            rng_seed=RNG_SEED,
        )
    )

    print("Best iter:", iter_best)
    print(stats_best)
    print(w_best)


if __name__ == "__main__":
    main()
