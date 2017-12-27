#!/bin/bash

# Copyright 2013   (Authors: Bagher BabaAli, Daniel Povey, Arnab Ghoshal)
#           2014   Brno University of Technology (Author: Karel Vesely)
#           2016   (Modified by: Sean Robertson)
# Apache 2.0.

# sdrobert: don't need any language modelling stuff. Also, utt2spk and spk2utt
# are now dummy maps (utt -> utt) because we don't want any speaker dependent
# jazz going on

if [ $# -ne 1 ]; then
   echo "Argument should be the Timit directory, see ../run.sh for example."
   exit 1;
fi

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

. ./path.sh # Needed for KALDI_ROOT
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

[ -f $conf/test_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f $conf/dev_spk.list ] || error_exit "$PROG: dev-set speaker list not found.";

# First check if the train & test directories exist (these can either be upper-
# or lower-cased
if [ ! -d $*/TRAIN -o ! -d $*/TEST ] && [ ! -d $*/train -o ! -d $*/test ]; then
  echo "timit_data_prep.sh: Spot check of command line argument failed"
  echo "Command line argument must be absolute pathname to TIMIT directory"
  echo "with name like /export/corpora5/LDC/LDC93S1/timit/TIMIT"
  exit 1;
fi

# Now check what case the directory structure is
uppercased=false
train_dir=train
test_dir=test
if [ -d $*/TRAIN ]; then
  uppercased=true
  train_dir=TRAIN
  test_dir=TEST
fi

tmpdir=$(mktemp -d /tmp/kaldi.XXXX);
trap 'rm -rf "$tmpdir"' EXIT

# Get the list of speakers. The list of speakers in the 24-speaker core test
# set and the 50-speaker development set must be supplied to the script. All
# speakers in the 'train' directory are used for training.
if $uppercased; then
  tr '[:lower:]' '[:upper:]' < $conf/dev_spk.list > $tmpdir/dev_spk
  tr '[:lower:]' '[:upper:]' < $conf/test_spk.list > $tmpdir/test_spk
  ls -d "$*"/TRAIN/DR*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
else
  tr '[:upper:]' '[:lower:]' < $conf/dev_spk.list > $tmpdir/dev_spk
  tr '[:upper:]' '[:lower:]' < $conf/test_spk.list > $tmpdir/test_spk
  ls -d "$*"/train/dr*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
fi

cd $dir
echo -n > $tmpdir/phndump
for x in train dev test; do
  # First, find the list of audio files (use only si & sx utterances).
  # Note: train & test sets are under different directories, but doing find on
  # both and grepping for the speakers will work correctly.
  find $*/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.WAV' \
    | grep -f $tmpdir/${x}_spk > ${x}_sph.flist

  sed -e 's:.*/\(.*\)/\(.*\).WAV$:\1_\2:i' ${x}_sph.flist \
    > $tmpdir/${x}_sph.uttids
  paste $tmpdir/${x}_sph.uttids ${x}_sph.flist \
    | sort -k1,1 > ${x}_sph.scp

  cat ${x}_sph.scp | awk '{print $1}' > ${x}.uttids

  # Now, Convert the transcripts into our format (no normalization yet)
  # Get the transcripts: each line of the output contains an utterance
  # ID followed by the transcript.
  find $*/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.PHN' \
    | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_phn.flist
  sed -e 's:.*/\(.*\)/\(.*\).PHN$:\1_\2:i' $tmpdir/${x}_phn.flist \
    > $tmpdir/${x}_phn.uttids
  while read line; do
    [ -f $line ] || error_exit "Cannot find transcription file '$line'";
    cut -f3 -d' ' "$line" | tr '\n' ' ' | sed -e 's: *$:\n:'
  done < $tmpdir/${x}_phn.flist > $tmpdir/${x}_phn.trans
  paste $tmpdir/${x}_phn.uttids $tmpdir/${x}_phn.trans \
    | sort -k1,1 > ${x}.text

  # sdrobert: extract phone segments. Because we are going to later obscure the
  # length of the target phone by cutting all segments to a fixed length, we
  # have to pad liberally to either side of each phone. A center point within
  # the segment is also recorded in case we have to cut those bounds because
  # they exceed the recording width. .32 seconds is chosen because only about
  # 34 phone instances exceed twice this value
  # note that I'm reusing the ${tmpdir}/${x}_phn.uttids file for my own
  # purposes and clear it
  echo -n "" > $tmpdir/${x}_phn.uttids
  echo -n "" > $tmpdir/${x}_phn.segments
  echo -n "" > $tmpdir/${x}_phn.centers
  echo -n "" > $tmpdir/${x}_phn.text
  while read line; do
    [ -f $line ] || error_exit "Cannot find transcription file '$line'";
    recid=$(echo $line | sed -e 's:.*/\(.*\)/\(.*\).PHN$:\1_\2:i')
    tmp=$(awk "BEGIN { recid=\"${recid}\"; idx=0; }"'{
      idx=idx+1;
      lo=$1/16000 - .32;
      hi=$2/16000 + .32;
      cent=(hi - lo)/2;
      phn=$3;
      if (lo < 0) { lo = 0; }
      printf "%s_%03i %s %f %f %s %f\n", recid, idx, recid, lo, hi, phn, cent;
    }' "$line")
    echo -e "$tmp" | cut -d ' ' -f 1 >> $tmpdir/${x}_phn.uttids
    echo -e "$tmp" | cut -d ' ' -f 1-4 >> $tmpdir/${x}_phn.segments
    echo -e "$tmp" | cut -d ' ' -f 1,5 >> $tmpdir/${x}_phn.text
    echo -e "$tmp" | cut -d ' ' -f 1,6 >> $tmpdir/${x}_phn.centers
  done < $tmpdir/${x}_phn.flist

  sort -k 1,1 $tmpdir/${x}_phn.centers > ${x}_phn.centers
  sort -k 1,1 $tmpdir/${x}_phn.uttids > ${x}_phn.uttids

  # sdrobert: since the center-phone guess task is not standardized, I don't
  # think there's historical precedent for reducing the phone set here.
  sort -k 1,1 $tmpdir/${x}_phn.text > ${x}_phn.text

  # sdrobert: we need to assign nonzero ids to all phones for ctc, so we dump
  # all the segments into this file, for now
  cut -d ' ' -f 2 ${x}_phn.text >> $tmpdir/phndump

  # Create wav.scp
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < ${x}_sph.scp \
    > ${x}_wav.scp

  # sdrobert: dummy utterance mappings
  paste -d ' ' $x.uttids $x.uttids > $x.utt2spk
  cp $x.utt2spk $x.spk2utt
  paste -d ' ' ${x}_phn.uttids ${x}_phn.uttids > ${x}_phn.utt2spk
  cp ${x}_phn.utt2spk ${x}_phn.spk2utt

  # sdrobert: use our duration file to make sure none of our phone boundaries
  # pass the end of the file
#  wav-to-duration scp:${x}_wav.scp ark,t:${x}_dur.ark || exit 1
#  echo -n > ${x}_phn.segments
#  while read line; do
#    uttid="${line% *}"
#    dur="${line#* }"
#    awk "BEGIN {dur=${dur};}"'$2 == "'"${uttid}"'" {
#      if ($4 > dur) { $4 = dur; }
#      print $1,$2,$3,$4;
#    }' $tmpdir/${x}_phn.segments | sort -k 1,1 >> ${x}_phn.segments
#  done < ${x}_dur.ark
done

# sdrobert: finish creating the phone to id map
sort $tmpdir/phndump | uniq | awk 'BEGIN {a=0}{print $1, a; a += 1}' \
  > phn_id.map

echo "Data preparation succeeded"
