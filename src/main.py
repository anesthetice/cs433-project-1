import numpy as np

import extra
import implementations as impl
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
    (y_train, x_train), (y_test, x_test) = get_pseudo_train_test_sets()

    w, loss = impl.reg_logistic_regression(
        y_train,
        x_train,
        lambda_=0.001,
        initial_w=np.ones(x_train.shape[1]),
        max_iters=300,
        gamma=0.3,
    )

    print(w)

    y_pred = -np.ones_like(y_test)
    y_prob = 1 / (1 + np.exp(-x_test @ w))
    y_pred[y_prob > 0.5] = +1

    extra.see_statistics(y_true=y_test, y_pred=y_pred)


if __name__ == "__main__":
    main()
