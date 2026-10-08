import helpers
import numpy as np


def main():
    x_train, x_test, y_train, train_ids, test_ids = helpers.load_data(
        data_path="./data/"
    )
    return

    print(f"x_train: shape={x_train.shape}, dtype={x_train.dtype}")
    print(f"y_train: shape={y_train.shape}, dtype={y_train.dtype}")
    print(f"train_ids: shape={train_ids.shape}, dtype={train_ids.dtype}")
    print()
    print(f"x_test: shape={x_test.shape}, dtype={x_test.dtype}")
    print(f"test_ids: shape={test_ids.shape}, dtype={test_ids.dtype}")
    print()
    print(
        f"is train_ids equivalent to `np.arange(x_train.shape[0])`? -> {np.all(train_ids == np.arange(x_train.shape[0]))}"
    )
    print()
    print(
        f"is test_ids equivalent to `x_train.shape[0] + np.arange(x_test.shape[0])`? -> {np.all(test_ids == x_train.shape[0] +  np.arange(x_test.shape[0]))}"
    )


if __name__ == "__main__":
    main()
