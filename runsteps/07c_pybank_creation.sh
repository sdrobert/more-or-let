#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

ref_rspecifier_dev=
ref_rspecifier_train=
ref_rspecifier_test=

source utils/parse_options.sh

if [ $# != 3 ]; then
  eecho "Usage: $0 <feat-name> <feat-root> <pybank-json>"
  echo "e.g. $0 fbank_40 data/41 conf/fbank_41.json"
  exit 1;
fi

feat_name="$1"
feat_root="$2"
pybank_json="$3"

iecho "${feat_name} creation"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT INT TERM

for x in dev test train; do
  iecho "Creating for $x partition"
  stepsext/make_pybank.sh \
    --pybank-json "${pybank_json}" \
    --nj $nj \
    --cmd "$cmd" \
    "${feat_root}/$x" \
    "exp/log/${feat_name}_creation" \
    "${tmpdir}/$x"
  mv "${feat_root}/$x/feats.scp" "${tmpdir}/$x/feats_${feat_name}.scp"
done

source runsteps/07b_feature_creation_common.sh \
  "${tmpdir}" "${feat_root}" "${feat_name}"

iecho "Made features for ${feat_name}"

rm -rf "${tmpdir}"
