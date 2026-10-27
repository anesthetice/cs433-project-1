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
        (y_train, x_train), (y_test, x_test), colidx_to_colname = (
            ex.get_pseudo_train_test_sets(
                target_features=target_features,
                train_to_test_ratio=0.8,
                rng_seed=rng_seed,
            )
        )

        (w_best, loss_best, iter_best), train_losses, test_losses = (
            ex.reg_logistic_regression_2(
                y_train,
                x_train,
                y_test,
                x_test,
                lambda_value=0.3,
                initial_w=None,
                gamma=0.15,
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

        for tr in thresholds:
            stats = ex.compute_statistics(
                y_true=y_test, y_pred=ex.logreg_predict(x_test, w_best, tr)
            )
            sensitivities.append(stats["sensitivity"])
            specificities.append(stats["specificity"])

        sensitivities = np.array(sensitivities)
        specificities = np.array(specificities)

        tr_intersect = thresholds[np.argmin(np.abs(sensitivities - specificities))]
        stats_intersect = ex.compute_statistics(
            y_true=y_test, y_pred=ex.logreg_predict(x_test, w_best, tr_intersect)
        )
        print(f"Threshold of {tr_intersect:.4f}")
        print(f"├─┄ TP = {stats_intersect['TP']}, TN = {stats_intersect['TN']}")
        print(f"├─┄ FP = {stats_intersect['FP']}, FN = {stats_intersect['FN']}")
        print(f"├─┄ sensitivity = {stats_intersect['sensitivity']:.2f}")
        print(f"└─┄ specificity = {stats_intersect['specificity']:.2f}\n")

        ax: plt.Axes = ax2
        ax.plot(thresholds, sensitivities, label="sensitivity")
        ax.plot(thresholds, specificities, label="specificity")
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
def _():
    IF2: list[dict] = [
        # General Health
        dict(colname="GENHLTH", type="ordcat", uvs=[7, 9], rnw=7),
        # Number of Days Physical Health Not Good
        dict(colname="PHYSHLTH", type="numerical", uvs=[77, 99], rnw=77),
        # Calculated variable for adults who have been told they have high blood pressure
        dict(colname="_RFHYPE5", type="ordcat", uvs=[9], rnw=9),
        # Currently Taking Blood Pressure Medication
        dict(colname="BPMEDS", type="ordcat", uvs=[7, 9], rnw=7),
        # Ever had a stroke?
        dict(colname="CVDSTRK3", type="ordcat", uvs=[7, 9], rnw=7),
        # Ever told you have diabetes
        dict(colname="DIABETE3", type="ordcat", uvs=[7, 9], rnw=7, replace=[(3, 5)]),
        # Education Level
        dict(colname="EDUCA", type="ordcat", uvs=[9], rnw=9),
        # Calculated variable for fourteen level age category, "AGE" by itself isn't here unfortunately.
        dict(colname="_AGEG5YR", type="ordcat", uvs=[14], rnw=14),
        # Calculated variable for reported height in meters.
        dict(colname="HTM4", type="numerical", uvs=[], rnw=9.99),
        # Calculated variable for reported weight in kilograms.
        dict(colname="WTKG3", type="numerical", uvs=[], rnw=999.99),
        # Calculated variable for four - categories of body mass index (bmi).
        dict(colname="_BMI5CAT", type="ordcat", uvs=[], rnw=7),
    ]

    # run1(IF2)
    return


if __name__ == "__main__":
    app.run()
