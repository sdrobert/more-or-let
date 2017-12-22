#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

source runsteps/xx_utility_funcs.sh

iecho "Timit s5 data prep"
resolve_timit
local/timit_data_prep.sh $TIMIT_DIR
local/timit_format_data.sh
