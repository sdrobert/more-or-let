#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

training_stage=adam

source utils/parse_options.sh

source runsteps/xx_utility_funcs.sh

if [ $# -ne 3 ] && [ $# -ne 4 ]; then
  eecho "Usage: $0 [options] <feat-name> <feat-root> <csv-path> [<model_formatter>]"
  eecho "e.g. $0 kaldi_123 data/123 exp/csv/kaldi_123.csv"
  exit 1;
fi

feat_name="$1"
feat_root="$2"
csv_path="$3"
model_formatter="$4"

if [ -z "${model_formatter}" ]; then
  model_formatter="exp/models/${feat_name}-{epoch:04d}.h5"
  iecho "model-formatter not set, using ${model_formatter}"
fi

iecho "Determining number of features and labels for training"

label_to_id_map_path="${feat_root}/train/phn_id.map"
num_feats=$(feat-to-dim "scp:${feat_root}/train/feats_${feat_name}.scp" -)
num_labels=$(echo 1 + $(
  cut -d' ' -f 2 "${label_to_id_map_path}" | sort | uniq | wc -l
) | bc)

iecho "Found ${num_feats} feats and ${num_labels} labels"

mkdir -p $(dirname "${csv_path}")
mkdir -p "${logdir}"
mkdir -p $(dirname "${model_formatter}") || wecho \
"Unable to create base directory of model-formatter. You should do this
manually"

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

$cmd "$logdir/train_${feat_name}_$(date +%s).log" \
  train-cnn-ctc \
    --num-feats=${num_feats} \
    --num-labels=${num_labels} \
    --verbose=${verbose} \
    --training-stage=${training_stage} \
    "${extra_args[@]}--model-formatter=${model_formatter}" \
    "--csv-path=${csv_path}" \
    "--config=${model_conf}" \
    "scp,s:${feat_root}/train/feats_${feat_name}.scp" \
    "ark,s:${feat_root}/train/text" \
    "${label_to_id_map_path}" \
    "scp,s:${feat_root}/dev/feats_${feat_name}.scp" \
    "ark,s:${feat_root}/dev/text"

iecho "Done ${training_stage} training for ${feat_name}"
