set -e

echo "Building dask_gdf"
conda build conda-recipes -c defaults -c conda-forge -c gpuopenanalytics/label/dev -c numba --python $PYTHON
