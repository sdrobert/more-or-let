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
compress=true
nj=10
cmd=$feat_cmd

source utils/parse_options.sh

set -e

# gateway to whatever features you want to generate
iecho "Feature creation"

# 40+1 filters, 10ms shift
mkdir -p data/41
rsync -r data/full/{dev,train,test} data/41

source runsteps/07a_kaldi_41_creation.sh

source runsteps/07c_pybank_creation.sh \
  --ref-rspecifier-train scp:data/41/train/feats_kaldi_41.scp \
  --ref-rspecifier-dev scp:data/41/dev/feats_kaldi_41.scp \
  --ref-rspecifier-test scp:data/41/test/feats_kaldi_41.scp \
  fbank_41 data/41 conf/fbank_41.json

source runsteps/07c_pybank_creation.sh \
  --ref-rspecifier-train scp:data/41/train/feats_kaldi_41.scp \
  --ref-rspecifier-dev scp:data/41/dev/feats_kaldi_41.scp \
  --ref-rspecifier-test scp:data/41/test/feats_kaldi_41.scp \
  sifbank_41 data/41 conf/sifbank_41.json

source runsteps/07c_pybank_creation.sh \
  --ref-rspecifier-train scp:data/41/train/feats_kaldi_41.scp \
  --ref-rspecifier-dev scp:data/41/dev/feats_kaldi_41.scp \
  --ref-rspecifier-test scp:data/41/test/feats_kaldi_41.scp \
  gbank_41 data/41 conf/gbank_41.json

source runsteps/07c_pybank_creation.sh \
  --ref-rspecifier-train scp:data/41/train/feats_kaldi_41.scp \
  --ref-rspecifier-dev scp:data/41/dev/feats_kaldi_41.scp \
  --ref-rspecifier-test scp:data/41/test/feats_kaldi_41.scp \
  sigbank_41 data/41 conf/sigbank_41.json

# 41 * 3 delta creation
mkdir -p data/123
rsync -r data/full/{dev,train,test} data/123

source runsteps/07d_delta_creation.sh data/41 kaldi_41 data/123 kaldi_123

source runsteps/07d_delta_creation.sh \
  --ref-rspecifier-train scp:data/123/train/feats_kaldi_123.scp \
  --ref-rspecifier-dev scp:data/123/dev/feats_kaldi_123.scp \
  --ref-rspecifier-test scp:data/123/test/feats_kaldi_123.scp \
  data/41 fbank_41 data/123 fbank_123

source runsteps/07d_delta_creation.sh \
  --ref-rspecifier-train scp:data/123/train/feats_kaldi_123.scp \
  --ref-rspecifier-dev scp:data/123/dev/feats_kaldi_123.scp \
  --ref-rspecifier-test scp:data/123/test/feats_kaldi_123.scp \
  data/41 sifbank_41 data/123 sifbank_123

source runsteps/07d_delta_creation.sh \
  --ref-rspecifier-train scp:data/123/train/feats_kaldi_123.scp \
  --ref-rspecifier-dev scp:data/123/dev/feats_kaldi_123.scp \
  --ref-rspecifier-test scp:data/123/test/feats_kaldi_123.scp \
  data/41 gbank_41 data/123 gbank_123

source runsteps/07d_delta_creation.sh \
  --ref-rspecifier-train scp:data/123/train/feats_kaldi_123.scp \
  --ref-rspecifier-dev scp:data/123/dev/feats_kaldi_123.scp \
  --ref-rspecifier-test scp:data/123/test/feats_kaldi_123.scp \
  data/41 sigbank_41 data/123 sigbank_123

iecho "Done feature creation"
