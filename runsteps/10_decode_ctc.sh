#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

set -e

source runsteps/xx_utility_funcs.sh

logdir=exp/log/decode
cmd=$decode_cmd
model_conf=conf/model.conf
verbose=0
decode_dir=

source utils/parse_options.sh

if $PREPROCESS_ON_BATCH ; then
  num_feats=41
else
  num_feats=123
fi

#source runsteps/10a_decode_ctc.sh \
#  kaldi_${num_feats} data/${num_feats} exp/csv/kaldi_${num_feats}.csv exp/scores/${num_feats}
#source runsteps/10a_decode_ctc.sh \
#  fbank_${num_feats} data/${num_feats} exp/csv/fbank_${num_feats}.csv exp/scores/${num_feats}
#source runsteps/10a_decode_ctc.sh \
#  sifbank_${num_feats} data/${num_feats} exp/csv/sifbank_${num_feats}.csv exp/scores/${num_feats}
source runsteps/10a_decode_ctc.sh \
  gbank_${num_feats} data/${num_feats} exp/csv/gbank_${num_feats}.csv exp/scores/${num_feats}
source runsteps/10a_decode_ctc.sh \
  sigbank_${num_feats} data/${num_feats} exp/csv/sigbank_${num_feats}.csv exp/scores/${num_feats}
