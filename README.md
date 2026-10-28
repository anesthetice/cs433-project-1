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
