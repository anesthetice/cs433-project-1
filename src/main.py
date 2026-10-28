import numpy as np

import extra
import preprocessing
import helpers
import consts
import plot


def main():
    RNG_SEED = 1
    DATA_PATH = "./data/"
    SUB_SAMPLE = False

    print(f"RNG_SEED: {RNG_SEED}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"SUB_SAMPLE: {SUB_SAMPLE}")
    print()
    print("## Loading and preprocessing the data...")

    # We first load the data and run our preprocessing pipeline (all-in-one)
    x_train, y_train, x_test, colidx_to_colname = preprocessing.load_and_preprocess(
        target_features=consts.INTERESTING_FEATURES,
        data_path=DATA_PATH,
        sub_sample=SUB_SAMPLE,
    )
    x_train_old_shape = x_train.shape


    # We split-off a portion of our training set into a validation set
    (y_train, x_train), (y_valid, x_valid) = extra.train_test_split(
        y=y_train,
        x=x_train,
        train_to_test_ratio=0.8,
        rng_seed=RNG_SEED
    )

    print()
    print("Split-off a validation set from the training set:")
    print(f"{x_train_old_shape} -> {x_train.shape} + {x_valid.shape} (validation)")

    # Standardize with the training stats only, and add a bias column
    x_train, x_valid, x_test = extra.standardize_and_add_bias(x_train, x_valid, x_test)

    print("Standardized features and made a bias column:")
    print(f"{x_train.shape[1] - 1} -> {x_train.shape[1]} features")
    print()
    print("## Running L2-regularized logistic regression targetting the iteration with the best F1-score across thresholds")

    (w_best, tr_best, stats_best, iter_best), train_losses, valid_losses = (
        extra.reg_logistic_regression_f1(
            y_train,
            x_train,
            y_valid,
            x_valid,
            lambda_value=1e-5,
            initial_w=np.zeros(x_train.shape[1]),
            gamma=1.19,
            n_iters=300,
            rng_seed=RNG_SEED,
        )
    )

    print(
        f"\nBest iter n°{iter_best}: F1 = {stats_best['F1']} (threshold = {tr_best})"
    )

    figure = plot.plot_training_and_stats(
        train_losses=train_losses,
        valid_losses=valid_losses,
        y_valid=y_valid,
        x_valid=x_valid,
        w=w_best,
    )
    figure.savefig("plot.svg")

    print("Rendered and saved plot to `plot.svg` in the current working directory")
    print()
    print("## Saving predictions to `output.csv` in the current working directory")

    helpers.create_csv_submission(
        ids=np.arange(328135, 328135 + x_test.shape[0]),
        y_pred=extra.logreg_predict(x_test, w_best, tr_best),
        name="output.csv"
    )



if __name__ == "__main__":
    main()
