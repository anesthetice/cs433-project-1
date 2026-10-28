import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import extra as ex
    import consts
    import helpers

    from importlib import reload
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set_theme()
    return consts, ex, np, plt, reload


@app.cell
def _(consts, ex, reload):
    reload(ex)
    reload(consts)
    return


@app.cell
def _(ex, np, plt):
    def run1(
        target_features: list[dict],
        rng_seed: int = 1,
    ):
        (y_train, x_train), (y_test, x_test), colidx_to_colname = ex.get_pseudo_train_test_sets(
            target_features=target_features,
            train_to_test_ratio=0.8,
            rng_seed=rng_seed,
        )

        (w_best, loss_best, iter_best), train_losses, test_losses = ex.reg_logistic_regression_2(
            y_train,
            x_train,
            y_test,
            x_test,
            lambda_value=0.1,
            initial_w=None,
            gamma=0.2,
            n_iters=250,
            batch_size=8192,
            rng_seed=rng_seed,
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

        ax: plt.Axes = ax1
        ax.plot(
            np.arange(len(train_losses)),
            train_losses,
            ls="--",
            color="blue",
            label="training loss",
        )
        ax.plot(
            np.arange(len(test_losses)),
            test_losses,
            ls="--",
            color="purple",
            label="testing loss",
        )
        ax.legend()

        thresholds = np.linspace(0, 1, 201)
        sensitivities = []
        specificities = []
        precisions = []
        f1s = []

        for tr in thresholds:
            stats = ex.compute_statistics(
                y_true=y_test, y_pred=ex.logreg_predict(x_test, w_best, tr)
            )
            sensitivities.append(stats["sensitivity"])
            specificities.append(stats["specificity"])
            precisions.append(stats["precision"])
            f1s.append(stats["F1"])

        sensitivities = np.array(sensitivities)
        specificities = np.array(specificities)
        precisions = np.array(precisions)
        f1s = np.array(f1s)

        tr_intersect = thresholds[np.argmax(f1s)]
        stats_intersect = ex.compute_statistics(
            y_true=y_test, y_pred=ex.logreg_predict(x_test, w_best, tr_intersect)
        )
        print(f"Threshold of {tr_intersect:.4f}")
        print(f"├─┄ TP = {stats_intersect['TP']}, TN = {stats_intersect['TN']}")
        print(f"├─┄ FP = {stats_intersect['FP']}, FN = {stats_intersect['FN']}")
        print(f"├─┄ sensitivity = {stats_intersect['sensitivity']:.2f}")
        print(f"├─┄ specificity = {stats_intersect['specificity']:.2f}")
        print(f"├─┄ precision = {stats_intersect['precision']:.2f}")
        print(f"└─┄ f1 = {stats_intersect['F1']:.2f}\n")

        ax: plt.Axes = ax2
        ax.plot(thresholds, sensitivities, label="sensitivity")
        ax.plot(thresholds, specificities, label="specificity")
        ax.plot(thresholds, precisions, label="precision")
        ax.plot(thresholds, f1s, label="F1-score")
        ax.vlines(
            x=tr_intersect,
            ymin=0,
            ymax=1,
            ls="--",
            color="gray",
            label="good threshold",
        )
        ax.set_xlabel("threshold")
        ax.legend()

        fig.tight_layout()
        return fig, (
            w_best,
            sensitivities,
            specificities,
            tr_intersect,
            stats_intersect,
            colidx_to_colname,
        )

    return (run1,)


@app.cell
def _(consts, run1):
    def _():
        (
            fig,
            (
                w_best,
                sensitivities,
                specificities,
                tr_intersect,
                stats_intersect,
                colidx_to_colname,
            ),
        ) = run1(consts.INTERESTING_FEATURES)

        prev_colname = ""
        for weight, colname in zip(w_best, colidx_to_colname):
            if prev_colname != colname:
                print(f"\n{colname}")
                prev_colname = colname
            print(f"{weight:.3f}")

        return fig

    _()
    return


@app.cell
def _(ex, np, plt):
    def run2(
        target_features: list[dict],
        rng_seed: int = 1,
    ):
        (y_train, x_train), (y_test, x_test), colidx_to_colname = ex.get_pseudo_train_test_sets(
            target_features=target_features,
            train_to_test_ratio=0.8,
            rng_seed=rng_seed,
        )

        (w_best, tr_best, stats_best, iter_best), train_losses, test_losses = (
            ex.reg_logistic_regression_f1(
                y_train,
                x_train,
                y_test,
                x_test,
                lambda_value=1e-5,
                initial_w=None,
                gamma=0.20,
                n_iters=150,
                batch_size=8192,
                rng_seed=rng_seed,
            )
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

        ax: plt.Axes = ax1
        ax.plot(
            np.arange(len(train_losses)),
            train_losses,
            ls="--",
            color="blue",
            label="training loss",
        )
        ax.plot(
            np.arange(len(test_losses)),
            test_losses,
            ls="--",
            color="purple",
            label="testing loss",
        )
        ax.legend()

        thresholds = np.linspace(0, 1, 201)
        sensitivities = []
        specificities = []
        precisions = []
        f1s = []

        for tr in thresholds:
            stats = ex.compute_statistics(
                y_true=y_test, y_pred=ex.logreg_predict(x_test, w_best, tr)
            )
            sensitivities.append(stats["sensitivity"])
            specificities.append(stats["specificity"])
            precisions.append(stats["precision"])
            f1s.append(stats["F1"])

        sensitivities = np.array(sensitivities)
        specificities = np.array(specificities)
        precisions = np.array(precisions)
        f1s = np.array(f1s)

        tr_intersect = thresholds[np.argmax(f1s)]
        stats_intersect = ex.compute_statistics(
            y_true=y_test, y_pred=ex.logreg_predict(x_test, w_best, tr_intersect)
        )
        print(f"Threshold of {tr_intersect:.4f}")
        print(f"├─┄ TP = {stats_intersect['TP']}, TN = {stats_intersect['TN']}")
        print(f"├─┄ FP = {stats_intersect['FP']}, FN = {stats_intersect['FN']}")
        print(f"├─┄ sensitivity = {stats_intersect['sensitivity']:.2f}")
        print(f"├─┄ specificity = {stats_intersect['specificity']:.2f}")
        print(f"├─┄ precision = {stats_intersect['precision']:.2f}")
        print(f"└─┄ f1 = {stats_intersect['F1']:.2f}\n")

        ax: plt.Axes = ax2
        ax.plot(thresholds, sensitivities, label="sensitivity")
        ax.plot(thresholds, specificities, label="specificity")
        ax.plot(thresholds, precisions, label="precision")
        ax.plot(thresholds, f1s, label="F1-score")
        ax.vlines(
            x=tr_intersect,
            ymin=0,
            ymax=1,
            ls="--",
            color="gray",
            label="good threshold",
        )
        ax.set_xlabel("threshold")
        ax.legend()

        fig.tight_layout()
        return fig, (
            w_best,
            sensitivities,
            specificities,
            tr_intersect,
            stats_intersect,
            colidx_to_colname,
        )

    return (run2,)


@app.cell
def _(consts, run2):
    def _():
        (
            fig,
            (
                w_best,
                sensitivities,
                specificities,
                tr_intersect,
                stats_intersect,
                colidx_to_colname,
            ),
        ) = run2(consts.INTERESTING_FEATURES)

        prev_colname = ""
        for weight, colname in zip(w_best, colidx_to_colname):
            if prev_colname != colname:
                print(f"\n{colname}")
                prev_colname = colname
            print(f"{weight:.3f}")

        return fig

    _()
    return


@app.cell
def _():
    from itertools import product

    return (product,)


@app.cell
def _(consts, ex, np, product):
    def run_find_best_f1_hyperparams(
        target_features: list[dict],
        rng_seed: int = 1,
    ):
        (y_train, x_train), (y_test, x_test), colidx_to_colname = ex.get_pseudo_train_test_sets(
            target_features=target_features,
            train_to_test_ratio=0.8,
            rng_seed=rng_seed,
        )

        lambda_values = [5e-1, 5e-3, 5e-5, 5e-7]
        gamma_values = [0.5, 0.25, 0.1]
        results = []

        for lambda_, gamma in product(lambda_values, gamma_values):
            (w_best, tr_best, stats_best, iter_best), train_losses, test_losses = (
                ex.reg_logistic_regression_f1(
                    y_train,
                    x_train,
                    y_test,
                    x_test,
                    lambda_value=lambda_,
                    initial_w=np.zeros(x_train.shape[1]),
                    gamma=gamma,
                    n_iters=200,
                    batch_size=8192,
                    rng_seed=rng_seed,
                )
            )

            results.append((lambda_, gamma, w_best, tr_best, stats_best, train_losses, test_losses))

        return results

    results = run_find_best_f1_hyperparams(consts.INTERESTING_FEATURES)
    return (results,)


@app.cell
def _(results):
    for lambda_, gamma, w_best, tr_best, stats_best, train_losses, test_losses in results:
        print(f"For lambda={lambda_:.1E}, and gamma={gamma:.2f}: best F1 = {stats_best['F1']:.3f}")
    return


@app.cell
def _(consts, ex):
    def run_knn(
        target_features: list[dict],
        rng_seed: int = 1,
    ):

        (y_train, x_train), (y_test, x_test), colidx_to_colname = ex.get_pseudo_train_test_sets(
            target_features=target_features,
            train_to_test_ratio=0.8,
            rng_seed=rng_seed,
        )

        y_pred = ex.predict_knn(y_train, x_train, x_test[0:500])

        stats = ex.compute_statistics(y_true=y_test[0:500], y_pred=y_pred)
        print(f"kNN with k=5")
        print(f"├─┄ TP = {stats['TP']}, TN = {stats['TN']}")
        print(f"├─┄ FP = {stats['FP']}, FN = {stats['FN']}")
        print(f"├─┄ sensitivity = {stats['sensitivity']:.2f}")
        print(f"├─┄ specificity = {stats['specificity']:.2f}")
        print(f"├─┄ precision = {stats['precision']:.2f}")
        print(f"└─┄ f1 = {stats['F1']:.2f}\n")

    run_knn(consts.INTERESTING_FEATURES)
    return


if __name__ == "__main__":
    app.run()
