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

from scipy.stats import friedmanchisquare, wilcoxon


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
        default='wer ~ C(feature) + C(si) + C(model)'
    )
    parser_descriptive = subparsers.add_parser(
        'descriptive',
        help='Calculate descriptive statistics, including mean, median, best, '
        'worst, and standard deviation'
    )
    parser_descriptive.add_argument(
        '--dependent-variable', default='wer',
        help='The dependent (measured) variable'
    )
    parser_descriptive.add_argument(
        '--independent-variables', nargs='*',
        default=['feature', 'si', 'partition'],
        help='A list of independent variables. Moments will be calculated for '
        'each unique value of their cross-product'
    )
    parser_friedman = subparsers.add_parser(
        'friedman',
        help='Run Friedman test'
    )
    parser_friedman.add_argument(
        '--dependent-variable', default='wer',
        help='The dependent (measured) variable'
    )
    parser_friedman.add_argument(
        '--independent-variables', nargs='+',
        default=['feature'],
        help='A list of independent variables. Samples for each element of '
        'the cross-product will be pulled to form a sample set'
    )
    parser_wilcoxon = subparsers.add_parser(
        'wilcoxon',
        help='Run Wilcoxon signed-rank test'
    )
    parser_wilcoxon.add_argument(
        '--dependent-variable', default='wer',
        help='The dependent (measured) variable'
    )
    parser_wilcoxon.add_argument(
        '--independent-variable', default='si',
        help='The independent variable. Must have two levels'
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
    elif args.type == 'descriptive':
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
            if mask is None:
                samples = df
            else:
                samples = df[mask]
            samples = samples[args.dependent_variable]
            if len(samples):
                print(
                    '{}\tN={:d} mean={:.02f} median={:.02f} std={:.02f} '
                    'min={:.02f} max={:.02f}'.format(
                        ','.join(intersection),
                        len(samples),
                        samples.mean(),
                        samples.median(),
                        samples.std(),
                        samples.min(),
                        samples.max(),
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
                sets.append(samples.values.flatten())
        print('Q={:.03f}, p={:.03f}'.format(*friedmanchisquare(*sets)))
    elif args.type == 'wilcoxon':
        categories = df[args.independent_variable].cat.categories
        if len(categories) != 2:
            raise ValueError('Independent variable must have two levels')
        sample_1 = df[df[args.independent_variable] == categories[0]]
        sample_1 = sample_1[args.dependent_variable].values.flatten()
        sample_2 = df[df[args.independent_variable] == categories[1]]
        sample_2 = sample_2[args.dependent_variable].values.flatten()
        print('W={:.03f}, p={:.03f}'.format(*wilcoxon(sample_1, sample_2)))
    else:
        print(
            'Successfully parsed data. To run some stats on them, specify '
            'a subcommand'
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
