#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

set -e

source runsteps/xx_utility_funcs.sh

logdir=exp/log/decode
cmd=$decode_cmd
model_conf=conf/model.conf
verbose=0
beam_width=
decode_dir=

source utils/parse_options.sh

source runsteps/10a_decode_ctc.sh \
  kaldi_123 data/123 exp/csv/kaldi_123.csv exp/scores/123
# source runsteps/10a_decode_ctc.sh \
#   fbank_123 data/123 exp/csv/fbank_123.csv exp/scores/123
# source runsteps/10a_decode_ctc.sh \
#   sifbank_123 data/123 exp/csv/sifbank_123.csv exp/scores/123
# source runsteps/10a_decode_ctc.sh \
#   gbank_123 data/123 exp/csv/gbank_123.csv exp/scores/123
# source runsteps/10a_decode_ctc.sh \
#   sigbank_123 data/123 exp/csv/sigbank_123.csv exp/scores/123
