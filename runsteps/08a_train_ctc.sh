#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

training_stage=adam
model_formatter=
train_formatter=
verbose=0

source utils/parse_options.sh

source runsteps/xx_utility_funcs.sh

if [ $# != 3 ]; then
  eecho "Usage: $0 [options] <feat-name> <feat-root> <exp-dir>"
  eecho "e.g. $0 kaldi_123 data/123 exp"
  exit 1;
fi

feat_name="$1"
feat_root="$2"
exp_dir="$3"

if [ -z "${model_formatter}" ]; then
  if [ "${training_stage}" = adam ]; then
    model_formatter="exp/model/${feat_name}-adam.TRIAL.h5"
  else
    model_formatter="exp/model/${feat_name}-{epoch:04d}.TRIAL.h5"
  fi
  iecho "model-formatter not set, using ${model_formatter}"
fi

if [ -z "${train_formatter}" ]; then
  if [ "${training_stage}" = adam ]; then
    train_formatter="exp/rng/${feat_name}-adam.TRIAL.pkl"
  else
    train_formatter="exp/rng/${feat_name}-{epoch:04d}.TRIAL.pkl"
  fi
  iecho "train-formatter not set, using ${train_formatter}"
fi

iecho "Determining number of features and labels for training"

label_to_id_map_path="${feat_root}/train/phn_id.map"
num_feats=$(feat-to-dim "scp:${feat_root}/train/feats_${feat_name}.scp" -)
num_labels=$(echo 1 + $(
  cut -d' ' -f 2 "${label_to_id_map_path}" | sort | uniq | wc -l
) | bc)

iecho "Found ${num_feats} feats and ${num_labels} labels"

mkdir -p "${exp_dir}/"{log,csv}
mkdir -p $(dirname "${model_formatter}") || wecho \
"Unable to create base directory of model-formatter. You should do this
manually"
mkdir -p $(dirname "${train_formatter}") || wecho \
"Unable to create base directory of train-formatter. You should do this
manually"

extra_args=()
if $PREPROCESS_ON_BATCH ; then
  extra_args=(
    "--delta-order=2"
    "--cmvn-rxfilename=${feat_root}/train/cmvn_${feat_name}.kdt"
    ""
  )
  # the last "" allows us to shove this into another arg's quotations without
  # ruining the behaviour of the last arg
fi

iecho "${training_stage} training for ${feat_name}"

$cmd TRIAL=1:${NUM_TRIALS} "${exp_dir}/log/train_${feat_name}_${training_stage}.TRIAL.log" \
  train-cnn-ctc \
    --num-feats=${num_feats} \
    --num-labels=${num_labels} \
    --verbose=${verbose} \
    --training-stage=${training_stage} \
    "--train-formatter=${train_formatter}" \
    "${extra_args[@]}--model-formatter=${model_formatter}" \
    "--csv-path=${exp_dir}/csv/${feat_name}.TRIAL.csv" \
    "--config=${model_conf}" \
    --weight-seed=\$\(\( 1000 + TRIAL \)\) \
    --train-seed=\$\(\( 2000 + TRIAL \)\) \
    "scp,s:${feat_root}/train/feats_${feat_name}.scp" \
    "ark,s:${feat_root}/train/text" \
    "${label_to_id_map_path}" \
    "scp,s:${feat_root}/dev/feats_${feat_name}.scp" \
    "ark,s:${feat_root}/dev/text" \|\| exit 1

iecho "Done ${training_stage} training for ${feat_name}"
