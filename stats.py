#! /usr/bin/env python

# Copyright 2018 Sean Robertson

from __future__ import division
from __future__ import print_function

import sys

from argparse import ArgumentParser
from re import compile

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main(args=None):
    '''Run some tests of significance on the output of results'''
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument(
        'in_file', nargs='?', default=None,
        help='Where to read RESULTS output from. If unset, reads from stdin')
    parser.add_argument('--power', type=float, default=.05)
    parser.add_argument(
        '--path-regex',
        help='Each line is matched against this expression. Groups are '
        'considered random variables',
        type=compile,
        default=compile(
            r'^%WER (?P<wer>[\d\.]+) .*/(?P<si>(si)?)(?P<feature>[^_]+)_\d+\.'
            r'?P<seed>\d+)/decode_(?P<partition>[^/]+)/.*')
    )
    parser.add_argument(
        '--continuous',
        default='wer',
        help='comma-delimited list of continuous (float) variables'
    )
    parser.add_argument(
        '--regression-formula',
        default='wer ~ C(feature) + C(si) + C(partition)'
    )
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args(args)
    if args.in_file:
        in_file = open(args.in_file)
    else:
        in_file = sys.stdin
    samples = dict()
    for line_no, line in enumerate(in_file):
        line = line.strip()
        match = args.path_regex.match(line)
        if match is None:
            if args.verbose:
                print(
                    'line {} ("{}") was unable to match pattern "{}"'.format(
                        line_no + 1, line, args.path_regex.pattern
                    ),
                    file=sys.stderr)
            continue
        match_dict = match.groupdict()
        for variable, value in match_dict.items():
            samples.setdefault(variable, []).append(value)
    Ns = tuple(len(x) for x in samples.values())
    if not len(Ns):
        print('No samples!', file=sys.stderr)
        return 1
    N = Ns[0]
    if not all(N == Np for Np in Ns):
        print('Not all samples have the same variables!', file=sys.stderr)
        return 1
    df = pd.DataFrame(samples, dtype='category')
    for variable in args.continuous.split(','):
        variable = variable.strip()
        if not variable:
            continue
        df[variable] = df[variable].astype(float)
    model = smf.ols(args.regression_formula, data=df).fit()
    print(sm.stats.anova_lm(model, type=2))
    print(model.summary())
    return 0


if __name__ == '__main__':
    sys.exit(main())
