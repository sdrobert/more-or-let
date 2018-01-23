#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

# Make an additional set of features from existing features by adding deltas
# in-feat-root is expected to have subdirs dev, train, test
# and files dev/feats_${name}.scp train/feats_${name}.scp
# test/feats_${name}.scp
# (a follow-up to 07c_pybank_creation)

ref_rspecifier_dev=
ref_rspecifier_train=
ref_rspecifier_test=
delta_order=2
delta_window=2

source utils/parse_options.sh

if [ $# != 4 ]; then
  eecho \
"Usage: $0 [options] <in-feat-root> <in-feat-name> <out-feat-root>
   <out-feat-name>"
  echo \
"e.g. $0 --delta-order 2 data/41 kaldi_41 data/123 kaldi_123"
  exit 1
fi
in_feat_root="$1"
in_feat_name="$2"
out_feat_root="$3"
out_feat_name="$4"

iecho "${out_feat_name} creation"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT INT TERM

for x in train dev test; do
  mkdir -p "${tmpdir}/$x"
  add-deltas \
    --verbose=${verbose} \
    --delta-order=${delta_order} \
    --delta-window=${delta_window} \
    "scp:${in_feat_root}/$x/feats_${in_feat_name}.scp" \
      "ark,scp:${tmpdir}/$x/feats_${out_feat_name}.ark,${tmpdir}/$x/feats_${out_feat_name}.scp"
done

source runsteps/07b_feature_creation_common.sh \
  "${tmpdir}" "${out_feat_root}" "${out_feat_name}"

rm -rf "$tmpdir"
