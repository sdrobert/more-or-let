#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

source runsteps/xx_utility_funcs.sh

if [ $# != 4 ]; then
  eecho "Usage: $0 [options] <feat-name> <feat-root> <model-path|csv_path> <score-dir>"
  eecho "e.g. $0 kaldi_123 data/123 exp/csv/kaldi_123.csv exp/scores/123"
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

iecho "Determining number of features and labels for decoding"

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
    "--batch-size=1"
    ""
  )
  # the last "" allows us to shove this into another arg's quotations without
  # ruining the behaviour of the last arg
  # also, performing delta ops on padded batches introduces a non-determinism
  # at test time, so we reduce the batch size to 1
fi

score_dir="$4"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT INT TERM

if [ -z "${decode_dir}" ]; then
  decode_dir="${tmpdir}"
fi

mkdir -p "${decode_dir}"
mkdir -p "${score_dir}"

iecho "Decoding ${feat_name}"

for x in test; do
  iecho "$x partition"
  mkdir -p "${score_dir}/${x}"
  mkdir -p "${decode_dir}/${x}"
  local/timit_norm_trans.pl \
    -i "${feat_root}/${x}/text" \
    -m "conf/phones.60-48-39.map" \
    -from 60 -to 39 \
    > "${decode_dir}/${x}/39_ref.txt"
  if [ $x = dev ]; then
    beam_widths=($(seq 1 20))
    beam_widths=(${beam_widths[*]} 100 1000)
    iecho "Trying beams ${beam_widths[*]}"
  else
    best_per=100
    best_beam=1
    for per_file in "${score_dir}/dev/per_${feat_name}."*.txt; do
      per=$(grep "Processed" $per_file | cut -d' ' -f 5 | cut -d'%' -f 1)
      beam=$(awk -F. '{print $(NF-1)}' <<< "${per_file}")
      iecho "PER for beam $beam was ${per}%"
      if [ $(bc -l <<< "$best_per > $per") = 1 ]; then
        best_beam=$beam
        best_per=$per
      fi
    done
    iecho "Using best beam $best_beam"
    beam_widths=(${best_beam})
  fi
  for beam_width in ${beam_widths[*]}; do
    iecho "Decoding $x partition's ${feat_name} with beam width ${beam_width}"
    $cmd $logdir/decode_${feat_name}.${beam_width}.log \
      decode-cnn-ctc \
        "scp,s:${feat_root}/$x/feats_${feat_name}.scp" \
        "ark,t,p:${decode_dir}/$x/60_hyp_${feat_name}.${beam_width}.txt" \
        "${label_to_id_map_path}" \
        "--verbose=${verbose}" \
        "--num-feats=${num_feats}" \
        "--num-labels=${num_labels}" \
        "--beam-width=${beam_width}" \
        "--model-path=${model_path}" \
        "${extra_args[@]}--config=${model_conf}"
    iecho "Decoded ${feat_name} in $x"
    iecho "Normalizing and computing PER for ${feat_name} in $x"
    local/timit_norm_trans.pl \
      -i "${decode_dir}/$x/60_hyp_${feat_name}.${beam_width}.txt" \
      -m "conf/phones.60-48-39.map" \
      -from 60 -to 39 \
      > "${decode_dir}/${x}/39_hyp_${feat_name}.${beam_width}.txt"
    iecho "Scoring ${x}"
    compute-error-rate --print-tables=true --strict=true \
      "ark:${decode_dir}/${x}/39_ref.txt" \
      "ark,t,p:${decode_dir}/${x}/39_hyp_${feat_name}.${beam_width}.txt" \
      "${score_dir}/${x}/per_${feat_name}.${beam_width}.txt"
    iecho "Normalized and computed PER for ${feat_name} in $x"
  done
done


iecho "Done decoding ${feat_name}"
