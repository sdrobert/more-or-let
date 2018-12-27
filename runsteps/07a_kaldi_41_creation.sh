#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

iecho "kaldi_41 creation"


ref_rspecifier_dev=
ref_rspecifier_train=
ref_rspecifier_test=
fbank_config=conf/kaldi_41.conf

source utils/parse_options.sh

tmpdir=$(mktemp -d)
trap "rm -rf '${tmpdir}'" EXIT

for x in test dev train; do
  iecho "Creating for $x partition"
  mkdir -p "${tmpdir}/$x"
  mkdir -p exp/log/kaldi_41_creation
  steps/make_fbank.sh \
    --compress ${compress} \
    --fbank-config "$fbank_config" \
    --nj $nj \
    --cmd "$cmd" \
    data/41/$x exp/log/kaldi_41_creation "${tmpdir}/$x"
  mv data/41/$x/feats.scp "${tmpdir}/$x/feats_kaldi_41.scp"
done

# sourcing keeps variables
source runsteps/07b_feature_creation_common.sh \
  "${tmpdir}" data/41 kaldi_41

iecho "Made features for kaldi_41"
