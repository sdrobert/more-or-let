#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

if [ $# != 4 ]; then
  eecho "Usage: $0 [options] <feat-name> <feat-root> <model-path|csv_path> <loss-dir>"
  eecho "e.g. $0 kaldi_123 data/123 exp/csv/kaldi_123.csv exp/losses/123"
  exit 1;
fi

feat_name="$1"
feat_root="$2"

if [ "${3##*.}" = "csv" ]; then
  model_path=$(
    find-best-model-from-log \
      "--verbose=${verbose}" \
      "--training-stage=sgd" \
      "$3"
  )
  iecho "Best model is ${model_path}"
else
  model_path="$3"
fi

iecho "Determining number of features and labels to calculate loss"

label_to_id_map_path="${feat_root}/train/phn_id.map"
num_feats=$(feat-to-dim "scp:${feat_root}/train/feats_${feat_name}.scp" -)
num_labels=$(echo 1 + $(
  cut -d' ' -f 2 "${label_to_id_map_path}" | sort | uniq | wc -l
) | bc)

iecho "Found ${num_feats} feats and ${num_labels} labels"

if $PREPROCESS_ON_BATCH ; then
  extra_args=(
    "--delta-order=2"
    "--cmvn-rxfilename=${feat_root}/train/cmvn_${feat_name}.kdt"
    ""
  )
  # the last "" allows us to shove this into another arg's quotations without
  # ruining the behaviour of the last arg
  # also, performing delta ops on padded batches introduces a non-determinism
  # at test time, so we reduce the batch size to 1
fi

loss_dir="$4"

iecho "Calculating loss for ${feat_name}"
for x in train dev test; do
  iecho "$x partition"
  mkdir -p "${loss_dir}/${x}"
  $cmd $logdir/get_loss_${x}_${feat_name}.log \
    compute-loss \
      "scp,s:${feat_root}/$x/feats_${feat_name}.scp" \
      "ark,s:${feat_root}/$x/text" \
      "${label_to_id_map_path}" \
      "--verbose=${verbose}" \
      "--num-feats=${num_feats}" \
      "--num-labels=${num_labels}" \
      "--model-path=${model_path}" \
      "${extra_args[@]}--config=${model_conf}" \> \
        "${loss_dir}/${x}/loss_${feat_name}.txt"
done
