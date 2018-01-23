#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

set -e

source runsteps/xx_utility_funcs.sh

logdir=exp/log/train
cmd=$train_cmd
model_conf=conf/model.conf
verbose=0
model_formatter=

source utils/parse_options.sh

if $PREPROCESS_ON_BATCH ; then
  num_feats=41
else
  num_feats=123
fi

iecho "Adam training"

# source runsteps/08a_train_ctc.sh \
#   --training-stage adam \
#   kaldi_${num_feats} data/${num_feats} exp/csv/kaldi_${num_feats}.csv
source runsteps/08a_train_ctc.sh \
  --training-stage adam \
  fbank_${num_feats} data/${num_feats} exp/csv/fbank_${num_feats}.csv
# source runsteps/08a_train_ctc.sh \
#   --training-stage adam \
#   sifbank_${num_feats} data/${num_feats} exp/csv/sifbank_${num_feats}.csv
# source runsteps/08a_train_ctc.sh \
#   --training-stage adam \
#   gbank_${num_feats} data/${num_feats} exp/csv/gbank_${num_feats}.csv
# source runsteps/08a_train_ctc.sh \
#   --training-stage adam \
#   sigbank_${num_feats} data/${num_feats} exp/csv/sigbank_${num_feats}.csv

iecho "Done Adam training"
