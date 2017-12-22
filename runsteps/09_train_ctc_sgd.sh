#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

set -e

source runsteps/xx_utility_funcs.sh

logdir=exp/log/train
cmd=$decode_cmd
model_conf=conf/model.conf
verbose=0
model_formatter=

source utils/parse_options.sh

iecho "SGD training"

source runsteps/08a_train_ctc.sh \
  --training-stage sgd \
  kaldi_123 data/123 exp/csv/kaldi_123.csv
source runsteps/08a_train_ctc.sh \
  --training-stage sgd \
  fbank_123 data/123 exp/csv/fbank_123.csv
source runsteps/08a_train_ctc.sh \
  --training-stage sgd \
  sifbank_123 data/123 exp/csv/sifbank_123.csv
source runsteps/08a_train_ctc.sh \
  --training-stage sgd \
  gbank_123 data/123 exp/csv/gbank_123.csv
source runsteps/08a_train_ctc.sh \
  --training-stage sgd \
  sigbank_123 data/123 exp/csv/sigbank_123.csv

iecho "Done SGD training"
