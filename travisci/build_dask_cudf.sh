set -e

echo "Building dask_cudf"
conda build conda-recipes -c rapidsai -c numba -c conda-forge -c defaults --python $PYTHON
