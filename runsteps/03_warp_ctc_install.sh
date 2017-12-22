#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

source runsteps/xx_utility_funcs.sh

iecho "warp-ctc install for tensorflow"
# FIXME(sdrobert): this isn't going to work properly on OS X
if $USE_CONDA; then
  INSTALL_DIR="${CONDA_PREFIX}"
elif $USE_VIRTUALENV; then
  INSTALL_DIR=$(readlink -f "${VIRTUALENV_DIR}")
else
   wecho \
"Installing warpctc to exp/warpctc. The install will break if you delete the
folder"
  INSTALL_DIR="`pwd`/exp/warpctc"
  mkdir -p "$INSTALL_DIR"
fi
tmp_dir=$(mktemp -d)
trap "rm -rf $tmp_dir" EXIT
pushd $tmp_dir
git clone https://github.com/baidu-research/warp-ctc.git
mkdir warp-ctc/build
cd warp-ctc/build
# FIXME
if true ; then
    iecho "Installing for GPU"
    cuda_code=$(nvcc -h | grep -oh 'sm_[[:digit:]][[:digit:]]' | sort | uniq | tail -n 3 | sed 's/^/code=/')
    cuda_arch=$(nvcc -h | grep -oh 'compute_[[:digit:]][[:digit:]]' | sort | uniq | tail -n 3 | sed 's/^/arch=/')
    nvcc_flags=$(paste <(echo "$cuda_arch") <(echo "$cuda_code") -d ',' | sed 's/^/-gencode /' | tr '\n' ' ')
    CXXFLAGS="$CXXFLAGS -D_GLIBCXX_USE_CXX11_ABI=0" cmake \
      -DWITH_GPU=ON \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
      -DCUDA_NVCC_FLAGS="$nvcc_flags" \
      ../
    # FIXME(sdrobert): do not disable openmp! Kiss of death for cpu
    make
    LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH}" ./test_cpu
    LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH}" ./test_gpu
    export CUDA_HOME="$(awk '$1 ~ /^CUDA_TOOLKIT_ROOT_DIR:/ {split($1, a, "="); print a[2]}' CMakeCache.txt)"
else
    iecho "Installing for CPU"
    cmake -DWITH_GPU=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ../
    make
    LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH}" ./test_cpu
fi
make install
cd ..
# git clone https://github.com/tensorflow/tensorflow.git
# export TENSORFLOW_SRC_PATH="$PWD/tensorflow"
# it appears that tensorflow has migrated whatever headers that were in its
# source path to the correct position. We can avoid pulling the source now (just
# pretend it's here)
# also, a couple of the unit tests aren't working right now, so we'll skip em
# https://github.com/baidu-research/warp-ctc/issues/59
cd tensorflow_binding
if [ "$(uname)" == "Darwin" ]; then
  TENSORFLOW_SRC_PATH="$PWD/.." \
  WARP_CTC_PATH="${INSTALL_DIR}/lib" \
  MACOSX_DEPLOYMENT_TARGET=$(sw_vers | awk '/^ProductVersion/ {print $2}') \
    python setup.py install
else
  TENSORFLOW_SRC_PATH="$PWD/.." \
  WARP_CTC_PATH="${INSTALL_DIR}/lib" \
  CFLAGS="$CFLAGS -D_GLIBCXX_USE_CXX11_ABI=0" \
    python setup.py install
fi
popd # has to come first, or python will clobber install with build dir
if ! python -c "import warpctc_tensorflow" ; then
    eecho "Could not import warpctc-tensorflow...removing"
    pip uninstall warpctc_tensorflow -y
    exit 1
fi
