# Dask GPU Dataframes

A partitioned gpu-backed dataframe, using Dask.

## Setup from source

Setup from source repo:

1.  Install dependencies into a new conda environment

        conda create -n dask-cudf \
           -c rapidsai -c numba -c conda-forge -c defaults \
           cudf dask cudatoolkit

2.  Activate conda environment:

        source activate dask-cudf

3.  Clone `dask_gdf` repo:

        git clone https://github.com/rapidsai/dask-cudf

4.  Install from source:

        cd dask-cudf
        pip install .

## Test

1.  Install `pytest`

        conda install pytest

2.  Run all tests:

        py.test dask_cudf

3. Or, run individual tests:

        py.test dask_cudf/tests/test_file.py

## Style

For style we use `black`, `isort`, and `flake8`.  These are available as
pre-commit hooks that will run every time you are about to commit code.

From the root directory of this project run the following:

```
pip install pre-commit
pre-commit install
```
