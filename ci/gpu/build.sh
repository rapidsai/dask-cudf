#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.
##############################################
# dask-cudf GPU build and test script for CI #
##############################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf

logger "Check versions..."
python --version
$CC --version
$CXX --version

logger "Setup new environment..."
conda install -c rapidsai -c rapidsai-nightly -c nvidia -c conda-forge -c defaults \
    cudf>=0.5 \
    dask>=0.19.0 \
    distributed>=1.23.0

logger "Python py.test for dask-cudf..."
cd $WORKSPACE
pip install -e .
py.test dask_cudf/ --cache-clear --junitxml=${WORKSPACE}/junit-dask-cudf.xml -v
