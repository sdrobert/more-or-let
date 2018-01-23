#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

source runsteps/xx_utility_funcs.sh

version=

source utils/parse_options.sh

iecho "Installing tensorflow"
if $TF_SOURCE_INSTALL ; then
  iecho "Compiling tensorflow from source"
  if ! which bazel > /dev/null; then
    eecho \
"'bazel' command cannot be found. If you want to compile tensorflow from source,
 make sure this is available. See
 https://bazel.build/versions/master/docs/install.html"
    exit 1
  fi
  tmp_dir=$(mktemp -d)
  trap "rm -rf $tmp_dir" EXIT TERM INT
  pushd $tmp_dir
  git clone https://github.com/tensorflow/tensorflow
  [ -z "$version" ] || git checkout tags/v${version}
  cd tensorflow
  PYTHON_BIN_PATH="$(which python)" \
    USE_DEFAULT_PYTHON_LIB_PATH=1 \
    TF_NEED_JEMALLOC=1 \
    TF_NEED_GCP=0 \
    TF_NEED_HDFS=0 \
    TF_ENABLE_XLA=0 \
    ./configure # user should figure out rest
  # assuming if the user configured cuda, we want to use it
  if bazel query '@local_config_cuda//crosstool:toolchain' > /dev/null 2>&1 ; then
    bazel build \
      --config=opt \
      --config=cuda \
      //tensorflow/tools/pip_package:build_pip_package
  else
    bazel build \
      --config=opt \
      //tensorflow/tools/pip_package:build_pip_package
  fi
  bazel-bin/tensorflow/tools/pip_package/build_pip_package \
    $PWD/tensorflow_pkg
  pip install --upgrade tensorflow_pkg/*.whl || exit 1
  popd
  if ! python -c "import tensorflow"; then
      eecho "CPU tensorflow could not be imported. Reverting"
      pip uninstall tensorflow -y
      exit 1
  fi
  exit 0
fi

if $USE_CONDA; then
  gpu_ok=true
  if [ -z "$version" ]; then
    gpu_name="tensorflow-gpu"
    cpu_name="tensorflow"
  else
    gpu_name="tensorflow-gpu=$version"
    cpu_name="tensorflow=$version"
  fi
  conda install $gpu_name -y || gpu_ok=false
  if $gpu_ok; then 
    if ! python -c "import tensorflow"; then
      gpu_ok=false
      conda remove tensorflow-gpu -y
    fi
  fi
  if ! $gpu_ok; then
    wecho \
'Unable to install tensorflow-gpu. Will try the cpu installation.
If you do have CUDA but this did not work, perhaps CUDA is not up
to tensorflow binary requirements. In this case, try building and
installing from source'
    conda install "$cpu_name" -y
    if ! python -c "import tensorflow"; then
      conda remove tensorflow -y
      exit 1
    fi
    iecho "CPU install successful"
  else
    iecho "GPU install successful"
  fi
else
  iecho "Installing tensorflow from PyPI"
  if [ -z "$version" ]; then
    gpu_name="tensorflow-gpu"
    cpu_name=tensorflow
  else
    gpu_name="tensorflow-gpu==$version"
    cpu_name="tensorflow==$version"
  fi
  gpu_ok=true
  pip install "$gpu_name" || gpu_ok=false
  if $gpu_ok; then
    if ! python -c "import tensorflow"; then
      pip uninstall tensorflow -y
      gpu_ok=false
    fi
  fi
  if ! $gpu_ok; then
    wecho \
'Unable to install tensorflow-gpu. Will try the cpu installation.
If you do have CUDA but this did not work, perhaps CUDA is not up
to tensorflow binary requirements. In this case, try building and
installing from source'
    pip install "$cpu_name"
    if ! python -c "import tensorflow"; then
      pip uninstall tensorflow -y
      exit 1
    fi
    iecho "CPU install successful"
  else
    iecho "GPU install successful"
  fi
fi
