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

iecho "Adam training"

# 123 feats (40+1 filters, + deltas + double deltas)
source runsteps/08a_train_ctc.sh \
  --training-stage adam \
  kaldi_123 data/123 exp/csv/kaldi_123.csv
# source runsteps/08a_train_ctc.sh \
#   --training-stage adam \
#   fbank_123 data/123 exp/csv/fbank_123.csv
# source runsteps/08a_train_ctc.sh \
#   --training-stage adam \
#   sifbank_123 data/123 exp/csv/sifbank_123.csv
# source runsteps/08a_train_ctc.sh \
#   --training-stage adam \
#   gbank_123 data/123 exp/csv/gbank_123.csv
# source runsteps/08a_train_ctc.sh \
#   --training-stage adam \
#   sigbank_123 data/123 exp/csv/sigbank_123.csv

iecho "Done Adam training"
