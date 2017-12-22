#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

# this is common to all feature creation
# we assume that data have been stored in a temporary directory
# we should have $tmpdir/dev $tmpdir/train $tmpdir/test with features
# referenced in script files $tmpdir/$x/feats_${name}.scp
# 1. If there's a reference feature set for a given partition:
#   a. Ensure the feature dimensions match the reference EXACTLY
#   b. Ensure the feature lengths match the reference. They will
#      be padded/truncated
# 2. Compute CMVN statistics for the train partition
# 3. Apply CMVN to each partition using train statistics
# 4. Put results in data dir. Script files will be saved to
#    $data/$x/feats_$name.scp, where $x is one of {train, dev, test}

# these variables should be defined from above
# tolerance=1
# strict=true
# pad_mode=edge
# ref_rspecifier_dev=
# ref_rspecifier_train=
# ref_rspecifier_test=
# norm_means=true
# norm_vars=true
# verbose=0
# compress=true

tmpdir="$1"
data="$2"
name="$3"

# train must come first!
for x in train dev test; do
  ref_var_name=ref_rspecifier_$x
  out_rspecifier="scp:${tmpdir}/$x/feats_${name}.scp"
  if [ ! -z "${!ref_var_name}" ]; then
    feat-to-dim --verbose=${verbose} \
      "${!ref_var_name}" "${tmpdir}/exp_dims"
    feat-to-dim --verbose=${verbose} \
      "$out_rspecifier" "${tmpdir}/act_dims"
    if ! cmp "${tmpdir}/exp_dims" "${tmpdir}/act_dims"; then
      eecho \
"New features of dim $(head -n 1 """${tmpdir}/act_dims""") do not match
reference dims $(head -n 1 """${tmpdir}/exp_dims""")."
      return 1
    fi
    feat-to-len --verbose=${verbose} \
      "${!ref_var_name}" "ark:${tmpdir}/ref_len_${name}.ark"
    normalize-feat-lens \
      --verbose ${verbose} \
      --tolerance ${tolerance} \
      --strict ${strict} \
      --pad-mode ${pad_mode} \
      "${out_rspecifier}" "ark:${tmpdir}/ref_len_${name}.ark" \
      "ark:${tmpdir}/$x/normed_${name}.ark"
    out_rspecifier="ark:${tmpdir}/$x/normed_${name}.ark"
  fi
  if $norm_means || $norm_vars; then
    if [ $x = "train" ]; then
      # compute CMVN stats in temp dir. Once we apply them, we don't need them
      # any more. Note these are speaker independent
      compute-cmvn-stats \
        --verbose=${verbose} \
        "${out_rspecifier}" "${tmpdir}/cmvn_${name}.kdt"
    fi
    apply-cmvn \
      --verbose=${verbose} \
      --norm-means=${norm_means} \
      --norm-vars=${norm_vars} \
      "${tmpdir}/cmvn_${name}.kdt" "${out_rspecifier}" \
      "ark:${tmpdir}/$x/standard_${name}.ark"
    out_rspecifier="ark:${tmpdir}/$x/standard_${name}.ark"
  fi
  mkdir -p "${data}/$x"
  copy-feats \
    --verbose=${verbose} \
    --compress=${compress} \
    "${out_rspecifier}" \
    "ark,scp:${data}/$x/feats_${name}.ark,${data}/$x/feats_${name}.scp"
done
