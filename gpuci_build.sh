#!/bin/bash
set -e
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib
CC=/usr/bin/gcc
CXX=/usr/bin/g++
DASKGDF_REPO=https://github.com/Quansight/dask_gdf
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
rm -rf /home/jenkins/.conda/envs/daskgdf
conda create -n daskgdf python=${PYTHON_VERSION}
conda install -n daskgdf -y -c numba -c conda-forge -c defaults -c gpuopenanalytics/label/dev \
  numba=${NUMBA_VERSION} \
  numpy=${NUMPY_VERSION} \
  pandas=${PANDAS_VERSION} \
  pyarrow=${PYARROW_VERSION} \
  pytest \
  dask \
  pygdf=0.1.0a3


logger "Activate conda env..."
source activate daskgdf

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

logger "Clone daskgdf..."
rm -rf $WORKSPACE/daskgdf
git clone --recurse-submodules ${DASKGDF_REPO} $WORKSPACE/daskgdf


logger "Build daskgdf..."
cd $WORKSPACE
python setup.py install

logger "Check GPU usage..."
nvidia-smi

logger "Test daskgdf..."
py.test --cache-clear --junitxml=junit.xml --ignore=daskgdf -v
