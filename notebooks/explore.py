# This python file is a marimo notebook used to explore the dataset and test some things out.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from compression import zstd

    import marimo as mo
    import polars as pl
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    from helpers import load_data
    from consts import COLNAME_TO_IDX, IDX_TO_COLNAME

    return COLNAME_TO_IDX, load_data, mo, np, pl, plt, sns, zstd


@app.cell
def _(load_data):
    x_train, x_test, y_train, train_ids, test_ids = load_data(data_path="./data")
    return test_ids, train_ids, x_test, x_train, y_train


@app.cell
def _(test_ids, train_ids, x_test, x_train, y_train):
    print(f"""
    x_train : {x_train.shape}     | (values are either missing or a whole positive integer)
    y_train : {y_train.shape}     | (values are either -1 or +1)
    train_ids : {train_ids.shape}

    x_test : {x_test.shape}
    test_ids : {test_ids.shape}
    """)
    return


@app.cell
def _(test_ids, train_ids):
    print(train_ids[0:10], "...", train_ids[-5:])
    print(test_ids[0:10])
    return


@app.cell
def _(mo, pl, zstd):
    x_test_df = pl.read_csv(zstd.open("./data/x_test.csv.zst"))
    mo.ui.table(
        data=x_test_df,
        max_columns=400,
    )
    return (x_test_df,)


@app.cell
def _(x_test_df):
    x_test_df["STREHAB1"].null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are quite the number of columns that aren't of interest to us, let's keep only the ones that are (or rather appear to be pertinent), for documentation, we will refer to the following:

    * The base info page: https://www.cdc.gov/brfss/annual_data/annual_2015.html
    * The description of the variables answered on the survey: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
        * This is where a lot of useless (to us at least) variables are, stuff like `IMONTH` (the interview month)
    * The calculated variables: https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_Calculated_Variables_Version4_08_10_17-508c.pdf
        * Many of these are however interesting to us.

    See `src/consts.py`
    """)
    return


@app.cell
def _(COLNAME_TO_IDX, np, x_train):
    np.isnan(x_train[:, COLNAME_TO_IDX["STREHAB1"]]).sum()
    return


@app.cell
def _(COLNAME_TO_IDX, np, plt, sns, x_train, y_train):
    def _():
        colname = "_BMI5CAT"
        coltype = "cat"

        colidx = COLNAME_TO_IDX[colname]
        col = x_train[:, colidx]

        finite_mask = np.isfinite(col)
        # col[~finite_mask] = -1

        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        ax: plt.Axes = ax1
        sns.histplot(
            x=col[finite_mask],
            hue=y_train[finite_mask],
            discrete=True,
            multiple="layer",
            # log_scale=(False, True)
        )

        return fig

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
