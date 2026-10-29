The repository for project 1 from the EPFL course cs-433

Group members:
- Elliot Currat (elliot.currat@epfl.ch)
- Anna Seres (anna.seres@epfl.ch)
- Alexandre Flusin (alexandre.flusin@epfl.ch)

> [!IMPORTANT]
> The `implementations.py` file is located in the `src/` directory.


## Getting Started

If you have [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed, there isn't really a need to do anything, for instance when you invoke the following command from within the directory: `uv run src/example.py`, `uv` will automatically create the appropriate virtual environment, assuming it doesn't already exist of course.

Otherwise, using a virtual environment, do the following:

``` sh
uv venv --python 3.14.8 .venv
source .venv/bin/activate
uv pip install numpy matplotlib seaborn
```

Once again, you should adapt the above to your specific virtual environment tool if you are not using `uv`.

You might also have to move the files located within the `src/` directory into the root directory for this repository.

## Generating the predictions

``` sh
# Just run the `main.py` script
uv run src/main.py
```

## Running Tests

``` sh
uv run pytest -v
```

## Linting

``` sh
uv run ruff check --fix .
uv run ruff format .
```

# Results

Results from running `src/main.py`:

```
Best iter n°295: F1 = 0.4408793009284544 (threshold = 0.2)
├─┄ TP = 3229, TN = 54208
├─┄ FP = 5558, FN = 2632
├─┄ sensitivity = 0.55
├─┄ specificity = 0.91
├─┄ precision = 0.37
└─┄ F1-score = 0.44
```


# Notes and Observations

The methodological reasoning is in our opinion quite flawed, our dataset cannot tell us who will develop
a heart condition in the future, only those that have had such a problem, survived, and therefore likely
made some big changes to their way of life after the fact. All of this to say that our model is picking
up on data that indicates someone has had (in the past) a heart attack, not really if they're likely 
to develop such a problem in the future.

A model that can only reliably predict if someone has had a heart attack or other serious cardiovascular
issue after said issue has occurred is not very useful... 

## Pre-processing

We load the data (more efficiently from a numpy-encoded file, that is created once if it does not exist yet, from the compressed `.csv` data files).

Given that a lot of the columns aren't very useful, we worked on an approach to make keeping only the interesting columns easier.

We describe a column (feature) we wish to keep using a structured dictionary,
as illustrated below:

``` py
INTERESTING_FEATURES: list[dict] = [
    # Number of Days Physical Health Not Good
    dict(colname="PHYSHLTH", type="numerical", uvs=[77, 99], replace=[(88, 0)]),

    # Have any health care coverage
    dict(colname="HLTHPLN1", type="ordcat", uvs=[7, 9]),

    # Multiple Health Care Professionals (not ordered categorical since values are not ordered: 1, 2, 0)
    dict(colname="PERSDOC2", type="cat", uvs=[7, 9]),

    # Could Not See Doctor Because of Cost
    dict(colname="MEDCOST", type="ordcat", uvs=[7, 9]),

    # Length of time since last routine checkup
    dict(colname="CHECKUP1", type="ordcat", uvs=[7, 9], replace=[(8, 5)]),

    # A lot more features not shown here, see `src/consts.py`
    # ...
]
```

We just have to write down which type of feature it is:

- "cat" for "categorical": no coherent order, must be encoded using one-hot approach.
- "ordcat" for "ordered categorical": has a coherent order (e.g., ascending BMI category), doesn't require one-hot encoding.
- "numerical": for things like the number of vegetables eaten per day.

And each of these dicts will be fed to a function that processes them into a matrix. Note that due to the large number of missing values, for all the above an extra column detailing if the value was missing or not (1 vs 0) is added, and median imputation is used for the missing values.

We also standardize, and add a bias ($w_0$) term.

Refer to `src/preprocess.py` and `src/extra.py` for more details.


## The Model

Logistic regression with L2-regularization, at every iteration we check various thresholds
and keep the model that overall managed to produce the maximum F1-score. Although,
we believe using a metric based on specificity and sensitivity would be better for our scenario,
nevertheless the competition is based off the F1-score.
