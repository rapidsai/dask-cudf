# Dask GPU Dataframes

A partitioned gpu-backed dataframe, using Dask.

## Setup from source

Setup from source repo:

1.  Install dependencies into a new conda environment

        conda install -n dask-gdf \
           -c numba -c conda-forge -c gpuopenanalytics/label/dev -c defaults \
           pygdf dask distributed cudatoolkit

2.  Activate conda environment:

        source activate dask-gdf

3.  Clone dask_gdf repo:

        git clone https://github.com/gpuopenanalytics/dask_gdf

4.  Install from source:

        cd dask_gdf
        pip install .

## Test

1.  Install `pytest`

        conda install pytest

2.  Run all tests:

        py.test dask_gdf

3. Or, run individual tests:

        pytest dask_gdf/tests/test_file.py
