#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

source runsteps/xx_utility_funcs.sh

beam_width=100

source utils/parse_options.sh

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  eecho "Usage: $0 [options] <feat-name> <feat-root> <exp-dir> [<model-path>]"
  eecho "e.g. $0 kaldi_123 data/123 exp"
  exit 1;
fi

feat_name="$1"
feat_root="$2"
exp_dir="$3"
model_path="$4"

if [ -z "${model_path}" ]; then
  mkdir -p "${exp_dir}/best_model_paths"
  run.pl TRIAL=1:${NUM_TRIALS} "${exp_dir}/log/find_best/TRIAL.log" \
  find-best-model-from-log \
    "--verbose=${verbose}" \
    "--training-stage=sgd" \
    "${exp_dir}/csv/${feat_name}.TRIAL.csv" \
    \> "${exp_dir}/best_model_paths/${feat_name}.TRIAL.txt"
  model_path_path="${exp_dir}/best_model_paths/${feat_name}.TRIAL.txt"
else
  if [ ${NUM_TRIALS} != 1 ] && [ -z "$(echo "${model_path}" | grep TRIAL)" ]; then
    eecho "More than one trial specified, but '${model_path}' does not contain
the 'TRIAL' keyword."
    exit 1
  fi
  tmpdir="$(mktemp -d)"
  trap "rm -rf '${tmpdir}'" EXIT
  run.pl TRIAL=1:${NUM_TRIALS} "${tmpdir}/TRIAL.log" \
    echo "${model_path}" \> "${tmpdir}/TRIAL.txt"
  model_path_path="${TMPDIR}/TRIAL.txt"
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
    ""
  )
  # the last "" allows us to shove this into another arg's quotations without
  # ruining the behaviour of the last arg
  # also, performing delta ops on padded batches introduces a non-determinism
  # at test time, so we reduce the batch size to 1
fi

iecho "Decoding ${feat_name}"

for x in dev test; do
  for t in $(seq 1 ${NUM_TRIALS}) ; do
    mkdir -p "${exp_dir}/${feat_name}.$t/decode_$x/scoring"
    cp "${feat_root}/$x/text" "${exp_dir}/${feat_name}.$t/decode_$x/scoring/ref_60phn.txt"
    local/timit_norm_trans.pl \
      -i "${exp_dir}/${feat_name}.$t/decode_$x/scoring/ref_60phn.txt" \
      -m conf/phones.60-48-39.map -from 60 \
      -to 39 > "${exp_dir}/${feat_name}.$t/decode_$x/scoring/ref_39phn.txt"
  done
  iecho "Decoding $x partition's ${feat_name} with beam width ${beam_width}"
  $cmd TRIAL=1:${NUM_TRIALS} "${exp_dir}/log/decode_$x_${feat_name}_${beam_width}.TRIAL.log" \
    decode-cnn-ctc \
      "scp,s:${feat_root}/$x/feats_${feat_name}.scp" \
      "ark,t,p:${exp_dir}/${feat_name}.TRIAL/decode_$x/scoring/${beam_width}_60phn.txt" \
      "${label_to_id_map_path}" \
      "--verbose=${verbose}" \
      "--num-feats=${num_feats}" \
      "--num-labels=${num_labels}" \
      "--beam-width=${beam_width}" \
      "--model-path=\$(cat ${model_path_path})" \
      "${extra_args[@]}--config=${model_conf}" \|\| exit 1
  iecho "Decoded ${feat_name} in $x"
  for t in $(seq 1 ${NUM_TRIALS}) ; do
    local/timit_norm_trans.pl \
      -i "${exp_dir}/${feat_name}.$t/decode_$x/scoring/${beam_width}_60phn.txt" \
      -m conf/phones.60-48-39.map -from 60 \
      -to 39 > "${exp_dir}/${feat_name}.$t/decode_$x/scoring/${beam_width}_39phn.txt"
  done
  run.pl TRIAL=1:${NUM_TRIALS} "${exp_dir}/${feat_name}.TRIAL/decode_$x/scoring/log/score_${beam_width}.log" \
    cat "${exp_dir}/${feat_name}.TRIAL/decode_$x/scoring/${beam_width}_39phn.txt" \| \
    compute-wer --text --mode=present \
    "ark:${exp_dir}/${feat_name}.TRIAL/decode_$x/scoring/ref_39phn.txt" \
    "ark,p:-" "&>" "${exp_dir}/${feat_name}.TRIAL/decode_$x/wer_${beam_width}"
done


iecho "Done decoding ${feat_name}"
