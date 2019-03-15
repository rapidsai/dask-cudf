#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh
export UPLOADFILE=`conda build conda-recipes -c nvidia -c rapidsai -c rapidsai-nightly -c numba -c conda-forge -c defaults --python $PYTHON --output`

set -e

SOURCE_BRANCH=master

test -e ${UPLOADFILE}

LABEL_OPTION="--label main"

# Restrict uploads to master branch
if [ ${GIT_BRANCH} != ${SOURCE_BRANCH} ]; then
  echo "Skipping upload"
  return 0
fi

if [ -z "$MY_UPLOAD_KEY" ]; then
  echo "No upload key"
  return 0
fi

echo "LABEL_OPTION=${LABEL_OPTION}"

echo "Upload"
echo ${UPLOADFILE}
anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${UPLOADFILE}
