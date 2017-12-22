#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

set -e

# global variables
CHECKPT=$1
ONE_STEP=false
export PROJECT_NAME="more-or-let"
export SCRIPT="$0"
export INFO_C="\e[35m"
export WARN_C="\e[93m"
export ERR_C="\e[91m"
export DEFT_C="\e[39m"
export USE_CONDA=true
export USE_VIRTUALENV=false
export CONDA_PYTHON_VER=3.6
export CONDA_ENV_NAME="${PROJECT_NAME}"
export VIRTUALENV_DIR="exp/venv" # the directory called with virtualenv
                                   # do not set to the parent. Could delete
                                   # everything!
export LOCK_FILE="exp/${PROJECT_NAME}.exclusivelock"
export LOCK_FD=200
export EXTRA_CONFIG_FNAME="extra_config.sh"
export TF_SOURCE_INSTALL=false # whether to install tensorflow from source
                               # or PyPI
export INSTALL_WARPCTC=false

source runsteps/00_preamble.sh

while : ; do
case $CHECKPT in

start)
iecho "Starting experiment from the beginning"
CHECKPT=0
;;

1)
runsteps/01_basics.sh
;;

2)
# For some reason, the switch from PoolAllocator to BFCAllocator caused OOM
# errors for me (12G GPU RAM). If this is not a problem for you, go ahead and
# remove this version flag
runsteps/02_tensorflow_install.sh --version 1.0.1
;;

3)
if $INSTALL_WARPCTC; then
  runsteps/03_warp_ctc_install.sh
else
  iecho "Skipping warp-ctc install. Will use built-in ctc"
fi
;;

4)
runsteps/04_keras_install.sh
;;

5)
runsteps/05_pydrobert_install.sh
;;

6)
runsteps/06_timit_data_prep.sh
;;

7)
runsteps/07_feature_creation.sh
;;

8)
runsteps/08_train_ctc_adam.sh
;;

9)
runsteps/09_train_ctc_sgd.sh
;;

10)
runsteps/10_decode_ctc.sh
;;

[0-9]*)
iecho "Done!"
exit 0
;;

*)
eecho "Invalid checkpoint passed: ${CHECKPT}"
exit 1
esac
checkpt
done
