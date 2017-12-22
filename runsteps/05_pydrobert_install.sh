#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

source runsteps/xx_utility_funcs.sh

if resolve_blas; then
  if ! resolve_swig; then
    eecho "Could not install or find an acceptable SWIG binary"
    exit 1
  fi
  pip install --upgrade \
    git+https://github.com/sdrobert/pydrobert-kaldi.git || \
      error_exit "Failed to install pydrobert-kaldi"
  if ! python -c "import pydrobert.kaldi" 2> /dev/null; then
    eecho "Could not import pydrobert-kaldi. Uninstalling"
    pip uninstall pydrobert-kaldi -y
    exit 1
  fi
elif $USE_CONDA; then
  wecho "Could not find blas library. Will download from anaconda cloud"
  conda install -c sdrobert pydrobert-kaldi || exit 1
fi

pip install --upgrade git+https://github.com/sdrobert/pydrobert-speech.git || \
  error_exit "Failed to install pydrobert-speech"
if ! python -c "import pydrobert.speech" 2> /dev/null; then
  eecho "Could not import pydrobert-speech. Uninstalling"
  pip uninstall pydrobert-speech -y
  exit 1
fi

pip install -e . || exit 1
