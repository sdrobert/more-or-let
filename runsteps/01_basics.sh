#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

source runsteps/xx_utility_funcs.sh

iecho "Setting up python environment"
if $USE_CONDA; then
  iecho "Using conda"
  if [ ! -z "$(conda env list | grep ${CONDA_ENV_NAME})" ]; then
    iecho "Deleting old ${CONDA_ENV_NAME} conda environment"
    source deactivate 2> /dev/null
    conda remove -n ${CONDA_ENV_NAME} --all -y 2> /dev/null || iecho "Nothing to remove"
  fi
  conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VER} statsmodels pandas numpy scipy h5py pyyaml six -y
  source activate ${CONDA_ENV_NAME}
else
  if $USE_VIRTUALENV; then
    iecho "Using virtualenv"
    if [ -d "${VIRTUALENV_DIR}" ]; then
      wecho "Deleting old virtualenv at ${VIRTUALENV_DIR}"
      rm -rf "${VIRTUALENV_DIR}"
    fi
    virtualenv --python "python${PYTHON_VER}" "${VIRTUALENV_DIR}"
    source "${VIRTUALENV_DIR}/bin/activate"
  else
    iecho "Using local python"
  fi
  pip install -U pip setuptools
  pip install numpy scipy h5py pyyaml six statsmodels pandas
fi
