#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

# preamble (to be sourced before we start stepping)

if [ -f "${EXTRA_CONFIG_FNAME}" ]; then
  source "${EXTRA_CONFIG_FNAME}"
fi

source runsteps/xx_utility_funcs.sh

# locking
function lock () {
  if [ "$(uname)" != "Darwin" ] ; then # need a good OS-X equivalent (TODO)
    eval "exec ${LOCK_FD}>${LOCK_FILE}"
    flock -n $LOCK_FD 2> /dev/null \
      && return 0 \
      || (\
        echo -e \
"Could not acquire lock. Either you are already running this script\n\
elsewhere or the program 'flock' is not installed. If you really know what\n\
you're doing, you can replace the code for this 'lock' function with\n\
'return 0' to try and run in parallel." && return 1)
  fi
}

# checkpointing
function checkpt () {
  ((CHECKPT++)) || true
  local old_chkpt="-1"
  if [ -f "exp/reached_checkpoints" ]; then
    # remove checkpoints after the current checkpoint
    local line_no=0
    local max_lines=$(wc -l "exp/reached_checkpoints" |\
      awk '{ print $1; }')
    while [ $old_chkpt -lt $CHECKPT ] && [ $line_no -le $max_lines ]; do
      ((line_no++)) || true
      local old_chkpt=$(head -n ${line_no} "exp/reached_checkpoints" |\
        tail -n 1)
    done
    ((line_no--)) || true  # old_chkpt either greater than or equal to chkpt;
                           # go back 1
    local tmp_name="$(mktemp)"
    head -n $line_no "exp/reached_checkpoints" > ${tmp_name}
    cp ${tmp_name} "exp/reached_checkpoints"
    rm ${tmp_name}
  fi
  echo $CHECKPT >> "exp/reached_checkpoints"
  if $ONE_STEP; then
    iecho "Finished step"
    exit 0
  fi
}
function print_checkpt () {
  echo -e \
"You can resume to the last checkpoint by calling '$SCRIPT'.\n\
To start from the beginning (and clear all progress), call\n\
'$SCRIPT clear'"
}

# define traps
function die () {
  local frame=0
  while caller $frame > /dev/null; do
    >&2 echo -e "${ERR_C}...From $(caller $frame)${DEFT_C}"
    ((frame++))
  done
  >&2 echo -e "${ERR_C}ERROR${DEFT_C}"
  print_checkpt
  exit 1
}
function sigint () {
  echo -e "${INFO_C}SIGINT${DEFT_C}"
  print_checkpt
  exit 0
}

# prep
mkdir -p exp || \
  error_exit "Could not make experiment directory"

source path.sh
source cmd.sh

lock || exit 1

if [ "${CHECKPT}" = "clear" ]; then
  iecho "Clearing everything... Are you sure?"
  read -p '["I am sure"/N]' response
  if [ "$response" != "I am sure" ]; then
    wecho 'That was not "I am sure" exactly. Aborting.'
    exit 0
  fi
  iecho "Well, okay, since you're sure"
  if $USE_CONDA; then
    conda remove -n ${CONDA_ENV_NAME} --all -y || true
  elif $USE_VIRTUALENV; then
    rm -rf "${VIRTUALENV_DIR}" 2> /dev/null || true
  fi
  rm -f exp/reached_checkpoints || true
  rm -rf exp/log/* || true
  rm -rf data/* || true
  exit 0
fi
if $USE_CONDA; then
  if ! command -v conda 1,2> /dev/null; then
    eecho "USE_CONDA was set to true, but could not find 'conda' command"
    exit 1
  fi
  if [ ! -z "$(conda env list | grep ${CONDA_ENV_NAME})" ]; then
    source activate ${CONDA_ENV_NAME}
  fi
elif $USE_VIRTUALENV; then
  if ! command -v virtualenv 1,2> /dev/null; then
    eecho \
"USE_VIRTUALENV was set to true, but could not find 'virtualenv' command"
    exit 1
  fi
  if [ -f "${VIRTUALENV_DIR}/bin/activate" ]; then
      source "${VIRTUALENV_DIR}/bin/activate"
  fi
else
  wecho "Using default python to install things. Highly not recommended"
fi

# resuming checkpoint
if [ -z "${CHECKPT}" ] || [ "${CHECKPT}" = "step" ]; then
  if [ "${CHECKPT}" = "step" ]; then
    ONE_STEP=true
  fi
  # look up last checkpoint. If none, start from beginnning
  if [ -f "exp/reached_checkpoints" ]; then
    CHECKPT=$(tail -n 1 "exp/reached_checkpoints")
  else
    CHECKPT="start"
  fi
elif [ "${CHECKPT}" = "redo" ]; then
  iecho "Will redo the last step (might not work)"
  if [ ! -f "exp/reached_checkpoints" ]; then
    eecho "Haven't started yet!'"
    exit 1
  fi
  CHECKPT=$(tail -n 2 "exp/reached_checkpoints" | head -n 1)
  ONE_STEP=true
else
  # a named checkpoint. Probably mucking with a dependency or something, so
  # only run for a step
  ONE_STEP=true
fi

# enable traps
trap die ERR
trap sigint SIGINT
