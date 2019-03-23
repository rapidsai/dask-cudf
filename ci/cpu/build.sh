#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.
###########################################
# dask-cudf CPU conda build script for CI #
###########################################
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

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# INSTALL - Install NVIDIA driver
################################################################################

logger "Install NVIDIA driver for CUDA $CUDA..."
apt-get update -q
DRIVER_VER="396.44-1"
LIBCUDA_VER="396"
if [ "$CUDA" == "10.0" ]; then
  DRIVER_VER="410.72-1"
  LIBCUDA_VER="410"
fi
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  cuda-drivers=${DRIVER_VER} libcuda1-${LIBCUDA_VER}

################################################################################
# BUILD - Conda package builds (conda deps: cudf <- dask-cudf)
################################################################################

logger "Build conda pkg for dask-cudf..."
source ci/cpu/dask-cudf/build_dask_cudf.sh

################################################################################
# UPLOAD - Conda packages
################################################################################

logger "Upload conda pkg..."
source ci/cpu/dask-cudf/upload-anaconda.sh
