# Copyright 2017 Sean Robertson

source runsteps/xx_utility_funcs.sh

version=

source utils/parse_options.sh

if [ -z "$version" ]; then
  min_version=2.0.9
else
  min_version=$version
fi

iecho "Keras install"
# keras versions take a bit of time to propagate to conda. Fall back to pip
# if a no-go
if $USE_CONDA && conda search --spec keras=$min_version >& /dev/null ; then
  gpu_ok=true
  conda list tensorflow | grep gpu > /dev/null || gpu_ok=false
  if [ -z "$version" ] && $gpu_ok; then
    pkg_name="keras-gpu"
  elif [ -z "$version" ]; then
    pkg_name="keras"
  elif $gpu_ok; then
    pkg_name="keras-gpu=$version"
  else
    pkg_name="keras=$version"
  fi
  conda install $pkg_name -y || exit 1
  if ! python -c 'import keras'; then
    conda remove $pkg_name -y
    exit 1
  fi
else
  if [ -z "$version" ]; then
    pkg_name="git+https://github.com/keras-team/keras.git"
  else
    pkg_name="keras==$version"
  fi
  pip install $pkg_name || exit 1
  if ! python -c 'import keras'; then
    pip uninstall keras -y
    exit 1
  fi
fi
iecho "Installed keras"
