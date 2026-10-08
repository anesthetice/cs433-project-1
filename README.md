The repository for project 1 from the EPFL course cs-433

## Building

If you have [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed, there isn't really a need to do anything, for instance when you invoke the following command from within the directory: `uv run src/example.py`, `uv` will automatically create the correct virtual environment if it doesn't already exist.

Otherwise, using a virtual environment, do the following:

``` sh
uv venv --python 3.14.8 .venv
source .venv/bin/activate
uv pip install numpy matplotlib seaborn
```

Once again, you should adapt the above to your specific virtual environment tool if you are not using `uv`.
