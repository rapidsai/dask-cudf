#!/usr/bin/env bash
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

logger "Building dask_cudf"
conda build conda/recipes/dask-cudf -c nvidia -c rapidsai -c rapidsai-nightly -c numba -c defaults -c conda-forge --python=$PYTHON
