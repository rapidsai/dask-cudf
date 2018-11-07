# Dask GPU Dataframes

A partitioned gpu-backed dataframe, using Dask.

## Setup from source

Setup from source repo:

1.  Install dependencies into a new conda environment

        conda install -n dask-cudf \
           -c numba -c conda-forge -c rapidsai -c defaults \
           pygdf dask distributed cudatoolkit

2.  Activate conda environment:

        source activate dask-cudf

3.  Clone dask_gdf repo:

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

        pytest dask_cudf/tests/test_file.py
