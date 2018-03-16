#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

set -e

source runsteps/xx_utility_funcs.sh

logdir=exp/log/losses
cmd=$decode_cmd
model_conf=conf/model.conf
verbose=0

source utils/parse_options.sh

if $PREPROCESS_ON_BATCH ; then
  num_feats=41
else
  num_feats=123
fi

source runsteps/11a_losses.sh \
  kaldi_${num_feats} data/${num_feats} exp/csv/kaldi_${num_feats}.csv exp/losses/${num_feats}
source runsteps/11a_losses.sh \
  fbank_${num_feats} data/${num_feats} exp/csv/fbank_${num_feats}.csv exp/losses/${num_feats}
source runsteps/11a_losses.sh \
  sifbank_${num_feats} data/${num_feats} exp/csv/sifbank_${num_feats}.csv exp/losses/${num_feats}
source runsteps/11a_losses.sh \
  gbank_${num_feats} data/${num_feats} exp/csv/gbank_${num_feats}.csv exp/losses/${num_feats}
source runsteps/11a_losses.sh \
  sigbank_${num_feats} data/${num_feats} exp/csv/sigbank_${num_feats}.csv exp/losses/${num_feats}
