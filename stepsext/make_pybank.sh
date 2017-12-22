#!/bin/bash

# Copyright 2012-2016  Karel Vesely  Johns Hopkins University (Author: Daniel Povey)
#           2017 (Modified by: Sean Robertson)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# sdrobert: this is a modification of make_fbank.sh to handle pydrobert-signal's
# feature creation

# Begin configuration section.
nj=1
cmd=run.pl
pybank_json='{}'
compress=true
pybank_conf=conf/pybank.conf
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> [<log-dir> [<feat-dir>] ]";
   echo "e.g.: $0 data/train exp/make_pybank/train pybank"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <feat-dir> defaults to <data-dir>/data"
   echo "Options: "
   echo "  --pybank-json                                    # path to pybank json (to build computer)"
   echo "  --pybank-conf                                    # additional arg file"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  pybankdir=$3
else
  pybankdir=$data/data
fi

# make $pybankdir an absolute pathname.
pybankdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $pybankdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $pybankdir || exit 1;
mkdir -p $logdir || exit 1;

scp=$data/wav.scp

required="$scp"
for f in $required; do
  if [ ! -f $f ]; then
    echo "make_pybank.sh: no such file $f"
    exit 1;
  fi
done

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

for n in $(seq $nj); do
  # the next command does nothing unless $pybankdir/storage/ exists, see
  # utils/create_data_link.pl for more info.
  utils/create_data_link.pl $pybankdir/raw_pybank_$name.$n.ark
done

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=""
  for n in $(seq $nj); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  $cmd JOB=1:$nj $logdir/make_pybank_${name}.JOB.log \
    extract-segments scp,p:$scp $logdir/segments.JOB ark:- \| \
      compute-feats-from-kaldi-tables \
          --verbose=2 \
          "--config=$pybank_conf" \
          ark:- ark:- "$pybank_json" \| \
        copy-feats --compress=$compress ark:- \
            "ark,scp:$pybankdir/raw_pybank_$name.JOB.ark,$pybankdir/raw_pybank_$name.JOB.scp" \
    || exit 1;

else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=""
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;

  $cmd JOB=1:$nj $logdir/make_pybank_${name}.JOB.log \
    compute-feats-from-kaldi-tables \
        --verbose=2 \
        "--config=$pybank_conf" \
        scp,p:$logdir/wav.JOB.scp ark:- "$pybank_json" \| \
      copy-feats --compress=$compress ark:- \
        "ark,scp:$pybankdir/raw_pybank_$name.JOB.ark,$pybankdir/raw_pybank_$name.JOB.scp" \
    || exit 1;

fi


if [ -f $logdir/.error.$name ]; then
  echo "Error producing pybank features for $name:"
  tail $logdir/make_pybank_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $pybankdir/raw_pybank_$name.$n.scp || exit 1;
done > $data/feats.scp

rm $logdir/wav.*.scp  $logdir/segments.* 2>/dev/null

nf=`cat $data/feats.scp | wc -l`
nu=`cat $data/utt2spk | wc -l`
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating filterbank features for $name"
