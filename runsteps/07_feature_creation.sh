#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

source runsteps/xx_utility_funcs.sh

# common args
tolerance=1
strict=true
pad_mode=edge
norm_means=true
norm_vars=true
verbose=0
compress=false
nj=10
cmd=$feat_cmd

source utils/parse_options.sh

set -e

# gateway to whatever features you want to generate
iecho "Feature creation"

# 40+1 filters, 10ms shift
mkdir -p data/41
rsync -r data/full/{dev,train,test} data/41

for feat in ${FEAT_NAMES}; do
  if [ "$feat" = kaldi ]; then
    source runsteps/07a_kaldi_41_creation.sh
  else
    source runsteps/07c_pybank_creation.sh \
      --ref-rspecifier-train scp:data/41/train/feats_kaldi_41.scp \
      --ref-rspecifier-dev scp:data/41/dev/feats_kaldi_41.scp \
      --ref-rspecifier-test scp:data/41/test/feats_kaldi_41.scp \
      ${feat}_41 data/41 conf/${feat}_41.json
  fi
done

if ! $PREPROCESS_ON_BATCH ; then
  # 41 * 3 delta creation
  mkdir -p data/123
  rsync -r data/full/{dev,train,test} data/123

  for feat in ${FEAT_NAMES}; do
    source runsteps/07d_delta_creation.sh \
      --norm-means false --norm-vars false \
      data/41 ${feat}_41 data/123 ${feat}_123
  done
fi

iecho "Done feature creation"
