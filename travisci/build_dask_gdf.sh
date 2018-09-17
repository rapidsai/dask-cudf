set -e

echo "Building dask_gdf"
conda build conda-recipes -c numba -c conda-forge -c gpuopenanalytics/label/dev -c defaults --python $PYTHON
