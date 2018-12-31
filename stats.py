#! /usr/bin/env python

# Copyright 2018 Sean Robertson

from __future__ import division
from __future__ import print_function

import sys

from argparse import ArgumentParser
from re import compile
from itertools import product

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.stats import friedmanchisquare


def main(args=None):
    '''Run some tests of significance on the output of results'''
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument(
        '--in-file', default=None,
        help='Where to read RESULTS output from. If unset, reads from stdin')
    parser.add_argument(
        '--path-regex',
        help='Each line is matched against this expression. Groups are '
        'considered random variables',
        type=compile,
        default=compile(
            r'^%WER (?P<wer>[\d\.]+) .*/(?P<si>(si)?)(?P<feature>[^_]+)_\d+\.'
            r'(?P<seed>\d+)/decode_(?P<partition>[^/]+)/.*')
    )
    parser.add_argument(
        '--continuous',
        default='wer',
        help='comma-delimited list of continuous (float) variables'
    )
    parser.add_argument('--verbose', action='store_true', default=False)
    subparsers = parser.add_subparsers(
        dest='type',
        help='Specify the type of analysis'
    )
    parser_manova = subparsers.add_parser(
        'manova',
        help='Perform Multivariate Analysis of Variance (MANOVA)'
    )
    parser_manova.add_argument(
        '--regression-formula',
        default='wer ~ C(feature) + C(si)'
    )
    parser_moments = subparsers.add_parser(
        'moments',
        help='Get first and second-order Gaussian moments'
    )
    parser_moments.add_argument(
        '--dependent-variable', default='wer',
        help='The dependent (measured) variable'
    )
    parser_moments.add_argument(
        '--independent-variables', nargs='+',
        default=['feature', 'si', 'partition'],
        help='A list of independent variables. Moments will be calculated for '
        'each unique value of their cross-product'
    )
    parser_moments = subparsers.add_parser(
        'friedman',
        help='Get first and second-order Gaussian moments'
    )
    parser_moments.add_argument(
        '--dependent-variable', default='wer',
        help='The dependent (measured) variable'
    )
    parser_moments.add_argument(
        '--independent-variables', nargs='+',
        default=['feature'],
        help='A list of independent variables. Samples for each element of '
        'the cross-product will be pulled to form a sample set'
    )
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
    if args.type == 'manova':
        model = smf.ols(args.regression_formula, data=df).fit()
        print(sm.stats.anova_lm(model, type=2))
        print(model.summary())
    elif args.type == 'moments':
        variables_levels = []
        print(
            ','.join(args.independent_variables) + '\t' +
            args.dependent_variable
        )
        print('===')
        for variable in args.independent_variables:
            variables_levels.append(sorted(df[variable].cat.categories))
        for intersection in product(*variables_levels):
            mask = None
            for variable, setting in zip(
                    args.independent_variables, intersection):
                if mask is None:
                    mask = df[variable] == setting
                else:
                    mask &= df[variable] == setting
            samples = df[mask][args.dependent_variable]
            if len(samples):
                print('{}\t{:.02f} ({:.02f})'.format(
                    ','.join(intersection),
                    samples.mean(),
                    samples.std()
                ))
    elif args.type == 'friedman':
        variables_levels = []
        for variable in args.independent_variables:
            variables_levels.append(sorted(df[variable].cat.categories))
        sets = []
        for intersection in product(*variables_levels):
            mask = None
            for variable, setting in zip(
                    args.independent_variables, intersection):
                if mask is None:
                    mask = df[variable] == setting
                else:
                    mask &= df[variable] == setting
            samples = df[mask][args.dependent_variable]
            if len(samples):
                print(intersection, len(samples))
                sets.append(samples.values.flatten())
        print('Friedman statistic: {:.03f}, p={:.03f}'.format(
            *friedmanchisquare(*sets)
        ))
    else:
        print(
            'Successfully parsed data. To run some stats on them, specify '
            'a subcommand'
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
