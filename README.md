# DASK_GDF

A partitioned gpu-backed dataframe, using dask.

## Setup

Setup from source repo:

1. Follow instructions in https://github.com/gpuopenanalytics/pygdf#setup to setup
   a conda environment for pygdf.
1. Activate pygdf environment: `source activate pygdf_dev`
1. Clone dask_gdf repo: `git clone https://github.com/gpuopenanalytics/dask_gdf path/to/dask_gdf`
1. `cd path/to/dask_gdf`
1. Install additional dependency: `conda install dask distributed`

Installing (Optional):
1. Run
   ```bash
   cd path/to/dask_gdf
   pip install .
   ```

Testing:
1. Ensure pytest is available: `conda install pytest`
1. `cd path/to/dask_gdf`
1. Run all tests with: `pytest dask_gdf`
1. Or, run individual tests with: `pytest dask_gdf/tests/test_file.py`
