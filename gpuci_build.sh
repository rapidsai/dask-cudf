#!/bin/bash
set -e
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib
CC=/usr/bin/gcc
CXX=/usr/bin/g++
DASKCUDF_REPO=https://github.com/rapidsai/dask_cudf
NUMBA_VERSION=0.40.0
NUMPY_VERSION=1.14.5
PANDAS_VERSION=0.20.3
PYTHON_VERSION=3.5
PYARROW_VERSION=0.10.0

function logger() {
  echo -e "\n>>>> $@\n"
}

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Create conda env..."
rm -rf /home/jenkins/.conda/envs/daskcudf
conda create -n daskcudf python=${PYTHON_VERSION}
conda install -n daskcudf -y -c rapidsai -c numba -c conda-forge -c defaults \
  numba=${NUMBA_VERSION} \
  numpy=${NUMPY_VERSION} \
  pandas=${PANDAS_VERSION} \
  pyarrow=${PYARROW_VERSION} \
  pytest \
  dask \
  cudf


logger "Activate conda env..."
source activate daskcudf

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

logger "Clone dask_cudf..."
rm -rf $WORKSPACE/daskcudf
git clone --recurse-submodules ${DASKCUDF_REPO} $WORKSPACE/daskcudf


logger "Build dask_cudf..."
cd $WORKSPACE
python setup.py install

logger "Check GPU usage..."
nvidia-smi

logger "Test dask_cudf..."
py.test --cache-clear --junitxml=junit.xml --ignore=daskcudf -v
