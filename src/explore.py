import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from helpers import load_data
    import implementations as impl

    import numpy as np
    import polars as pl
    import marimo as mo

    from compression import zstd

    return load_data, mo, pl, zstd


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
    for series in x_test_df.null_count().iter_columns():
        print(f"{series.name}: {series[0]}")
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
    """)
    return


if __name__ == "__main__":
    app.run()
