# more-or-let
Feature experiments with an end-to-end convolutional phone recognizer

## What is it?
A somewhat over-engineered Kaldi recipe for end-to-end phoneme recognition on
TIMIT. The model is based on that of Zhang et al. [1], with a few minor
modifications. The recipe tests a variety of time-frequency features in
using Kaldi's built-in features as well as some provided by
_pydrobert-speech_ [2].

## Getting started
1. Clone this repository into _kaldi/egs/timit/_.
2. Softlink the _steps_ and _utils_ folders from _../../wsj/s5_ into this
   directory.
3. Take a look at _run.sh_ and _cmd.sh_, modifying them as you see fit.
4. Call `./run.sh`
Run steps 1-5 involve setting up the python environment, including things like
installing tensorflow from source. I tend to just do this manually and skip to
step 6.

## License
This code falls under Apache License 2.0 (see _LICENSE_). _run.sh_, as well
as the files in _local_ are either inspired, copied, or heavily modified from
Kaldi [3]. Kaldi is licensed under Apache License 2.0. Its notice can be found
in _COPYING\_kaldi_.

## References
[1]: https://arxiv.org/abs/1701.02720
[2]: https://github.com/sdrobert/pydrobert-speech
[3]: http://kaldi-asr.org/
