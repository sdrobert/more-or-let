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

for feat in ${FEAT_NAMES} ; do
  source runsteps/10a_decode_ctc.sh \
    ${feat}_${num_feats} data/${num_feats} exp
done
