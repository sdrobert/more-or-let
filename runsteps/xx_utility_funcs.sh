#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

# Utility functions. All scripts can source this

# message printing
function iecho () {
  echo -e "${INFO_C}[$(caller)]INFO:${DEFT_C} "$*
}
function wecho () {
  echo -e "${WARN_C}[$(caller)]WARN:${DEFT_C} "$*
}
function eecho () {
  >&2 echo -e "${ERR_C}[$(caller)]ERROR:${DEFT_C} "$*
}


function error_exit () {
  # we duplicate eecho here because calling adds to the stack, misrepping callr
  >&2 echo -e "${ERR_C}[$(caller)]ERROR:${DEFT_C} "$*
  exit 1
}


# if we need the timit dir
function resolve_timit() {
  if [ -d "${TIMIT_DIR}/TIMIT" ]; then
    export TIMIT_DIR="${TIMIT_DIR}/TIMIT"
  fi
  if [ ! -d "${TIMIT_DIR}/TRAIN" ]; then
    eecho "TIMIT_DIR not set or set incorrectly. Export as path to TIMIT"
    exit 1
  else
    iecho "TIMIT_DIR set to $TIMIT_DIR"
  fi
}


# when building pydrobert-kaldi, we need some form of blas library. This block
# should handle it in most normal cases, though it's a bit fragile
function resolve_blas() {
  # check if set by user, and use that
  blas_specs=(
    ${OPENBLASROOT}
    ${ATLASROOT}
    ${MKLROOT}
    ${CLAPACKROOT}
    ${LAPACKEROOT}
    ${ACCELERATE}
    ${BLAS_LIBRARIES}
    ${BLAS_INCLUDES}
  )
  [ ${#blas_specs[@]} -gt 0 ] && return 0  # setup.py will error if >1 set
  # get it from the Kaldi configuration, if possible
  if [ -f "${KALDI_ROOT}/src/kaldi.mk" ]; then
      # If we're on OSX, try Accelerate
    if [ "`uname`" == "Darwin" ]; then
      export ACCELERATE=1
      return 0
    fi
    for blas_name in openblas mkl atlas clapack; do
      if grep -i "$blas_name" "${KALDI_ROOT}/src/kaldi.mk" > /dev/null 2>&1; then
        export BLAS_LIBRARIES=$(
          grep -i "${blas_name}libs = " "${KALDI_ROOT}/src/kaldi.mk" |
          cut -d' ' -f 3-)
        export BLAS_INCLUDES=$(
          grep -i "${blas_name}inc = " "${KALDI_ROOT}/src/kaldi.mk" |
          cut -d' ' -f 3-)
        return 0
      fi
    done
  fi
  eecho \
"Unable to find an appropriate BLAS library. Please set one of OPENBLASROOT,
MKLROOT, ATLASROOT, CLAPACKROOT, LAPACKEROOT, or ACCELERATE"
  return 1
}


# from
# https://stackoverflow.com/questions/4023830/how-to-compare-two-strings-in-dot-separated-version-format-in-bash
function vercomp () {
    if [[ $1 == $2 ]]
    then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++))
    do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++))
    do
        if [[ -z ${ver2[i]} ]]
        then
            # fill empty fields in ver2 with zeros
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]}))
        then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]}))
        then
            return 2
        fi
    done
    return 0
}


# when building pydrobert-kaldi, we need swig. This makes sure swig can be found
function resolve_swig() {
  if which swig > /dev/null; then
    swig_ver=$(swig -version | grep "Version" | awk '{print $3}')
    vercomp $swig_ver 3.0.8
    case $? in
      0) iecho "Acceptable SWIG version found"; return 0;;  # == 3.0.8
      1) iecho "Acceptable SWIG version found"; return 0;;  # > 3.0.8
      2) wecho "Swig version ($swig_ver) too low.";;  # < 3.0.8
    esac
  fi
  if $USE_CONDA; then
    iecho "Installing SWIG from Conda"
    conda install swig -y || return 1;
    iecho "Installed SWIG from Conda"
    return 0
  else
    iecho "Installing SWIG from source"
    pushd "$VIRTUALENV_DIR" || return 1;
    abs_venv_path="$(pwd -P)"
    popd || return 1;
    # virtualenv. Download and install into virtualenv's bin
    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT
    pushd $tmpdir || return 1
    wget https://github.com/swig/swig/archive/rel-3.0.8.tar.gz || (
      popd; return 1)
    if [ `uname` == Darwin ]; then
      [ "$(md5 -q rel-3.0.8.tar.gz)" = "9b5862b1d782b111d87fce9216a2d465" ] || (
        popd; return 1)
    else
      [ "$(md5sum rel-3.0.8.tar.gz | cut -d' ' -f 1)" = "9b5862b1d782b111d87fce9216a2d465" ] || (
        popd; return 1)
    fi
    tar -xf rel-3.0.8.tar.gz || (popd; return 1)
    cd swig-rel-3.0.8
    ./autogen.sh || (popd; return 1)
    ./configure --prefix="${abs_venv_path}"
    make || (popd; return 1)
    make install || (popd; return 1)
    (which swig | grep "${abs_venv_path}") || (popd; return 1)
    popd
    iecho "Installed SWIG from source"
    return 0
  fi
}
