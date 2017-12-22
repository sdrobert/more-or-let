#!/bin/bash

# Copyright 2013  (Author: Daniel Povey)
#           2016  (Modified by: Sean Robertson)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/train, etc.

. ./path.sh || exit 1;

echo "Preparing train, dev and test data"
srcdir=data/local/data

for x in train dev test; do
  mkdir -p data/full/$x
  cp $srcdir/${x}_wav.scp data/full/$x/wav.scp || exit 1;
  cp $srcdir/$x.text data/full/$x/text || exit 1;
  cp $srcdir/$x.spk2utt data/full/$x/spk2utt || exit 1;
  cp $srcdir/$x.utt2spk data/full/$x/utt2spk || exit 1;
  cp $srcdir/phn_id.map data/full/$x
  utils/validate_data_dir.sh --no-feats data/full/$x || exit 1;
  mkdir -p data/segs/$x
  cp $srcdir/${x}_wav.scp data/segs/$x/wav.scp || exit 1;
  cp $srcdir/${x}_phn.text data/segs/$x/text || exit 1;
  cp $srcdir/${x}_phn.spk2utt data/segs/$x/spk2utt || exit 1;
  cp $srcdir/${x}_phn.utt2spk data/segs/$x/utt2spk || exit 1;
  sort -k 1,1 $srcdir/${x}_phn.segments > data/segs/$x/segments || exit 1;
  cp $srcdir/phn_id.map data/segs/$x
  utils/validate_data_dir.sh --no-feats data/segs/$x || exit 1;
done
