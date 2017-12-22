'''Command line entry points for more-or-let'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

from collections import defaultdict
from csv import DictReader
from itertools import chain

import numpy as np

from pydrobert.kaldi.command_line import KaldiParser
from pydrobert.kaldi.command_line import kaldi_bool_arg_type
from pydrobert.kaldi.io import open as io_open
from pydrobert.kaldi.logging import kaldi_logger_decorator
from pydrobert.kaldi.logging import kaldi_vlog_level_cmd_decorator
from pydrobert.kaldi.logging import register_logger_for_kaldi
from pydrobert.mol.config import DecodeConfig
from pydrobert.mol.config import ModelConfig
from pydrobert.mol.config import TrainConfig
from pydrobert.mol.corpus import DecodeData
from pydrobert.mol.corpus import TrainData
from pydrobert.mol.corpus import ValidationData
from pydrobert.mol.util import edit_distance

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

__all__ = [
    'compute_per',
    'train_cnn_ctc',
    'decode_cnn_ctc',
    'find_best_model_from_log'
]


def _compute_per_parse_args(args, logger):
    parser = KaldiParser(
        description=compute_per.__doc__,
        add_verbose=True,
        logger=logger,
    )
    parser.add_argument(
        'ref_rspecifier', type='kaldi_rspecifier',
        help='Rspecifier pointing to reference (gold standard) transcriptions')
    parser.add_argument(
        'hyp_rspecifier', type='kaldi_wspecifier',
        help='Rspecifier pointing to hypothesis transcriptions')
    parser.add_argument(
        'out_path', nargs='?', default=None,
        help='Path to print results to. Default is stdout.')
    parser.add_argument(
        '--print-tables', type='kaldi_bool', default=False,
        help='If set, will print breakdown of insertions, deletions, and subs')
    parser.add_argument(
        '--strict', type='kaldi_bool', default=False,
        help='If set, missing utterances will cause an error')
    options = parser.parse_args(args)
    return options


@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def compute_per(args=None):
    '''Compute phone error rate between reference and hypothesis

    `ref_rspecifier` and `hyp_rspecifier` are expected to be ordered.
    If not, results may be incorrect.
    '''
    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(sys.argv[0])
    options = _compute_per_parse_args(args, logger)
    global_edit = 0
    global_token_count = 0
    global_sents = 0
    global_processed = 0
    inserts = defaultdict(lambda: 0)
    deletes = defaultdict(lambda: 0)
    subs = defaultdict(lambda: 0)
    totals = defaultdict(lambda: 0)

    def _err_on_utt_id(utt_id, missing_rxspecifier):
        msg = "Utterance '{}' absent in '{}'".format(
            utt_id, missing_rxspecifier)
        if options.strict:
            logger.error(msg)
            return 1
        else:
            logger.warning(msg)
            return 0
    with io_open(options.ref_rspecifier, 'tv') as ref_table, \
            io_open(options.hyp_rspecifier, 'tv') as hyp_table:
        while not ref_table.done() and not hyp_table.done():
            global_sents += 1
            if ref_table.key() > hyp_table.key():
                if _err_on_utt_id(hyp_table.key(), options.ref_rspecifier):
                    return 1
                hyp_table.move()
            elif hyp_table.key() > ref_table.key():
                if _err_on_utt_id(ref_table.key(), options.hyp_rspecifier):
                    return 1
                ref_table.move()
            else:
                logger.debug('Processing {}: ref [{}] hyp [{}]'.format(
                    ref_table.key(),
                    ' '.join(ref_table.value()),
                    ' '.join(hyp_table.value())))
                global_token_count += len(ref_table.value())
                res = edit_distance(
                    ref_table.value(), hyp_table.value(),
                    return_tables=options.print_tables
                )
                if options.print_tables:
                    global_edit += res[0]
                    for global_dict, utt_dict in zip(
                            (inserts, deletes, subs, totals), res[1:]):
                        for phone, count in utt_dict.items():
                            global_dict[phone] += count
                else:
                    global_edit += res
            global_processed += 1
            ref_table.move()
            hyp_table.move()
        while not ref_table.done():
            if _err_on_utt_id(ref_table.key(), options.hyp_rspecifier):
                return 1
            global_sents += 1
            ref_table.move()
        while not hyp_table.done():
            if _err_on_utt_id(hyp_table.key(), options.ref_rspecifier):
                return 1
            global_sents += 1
            hyp_table.move()
    if options.out_path is None:
        out_file = sys.stdout
    else:
        out_file = open(options.out_path, 'w')
    print(
        "Processed {}/{}. PER: {:.2f}%".format(
            global_processed, global_sents,
            global_edit / global_token_count * 100),
        file=out_file
    )
    if options.print_tables:
        print(
            "Total insertions: {}, deletions: {}, substitutions: {}".format(
                sum(inserts.values()), sum(deletes.values()),
                sum(subs.values())),
            file=out_file,
        )
        print("", file=out_file)
        phones = list(set(inserts) | set(deletes) | set(subs))
        phones.sort()
        phone_len = max(max(len(phone) for phone in phones), 5)
        max_count = max(
            chain(inserts.values(), deletes.values(), subs.values()))
        max_count_len = int(np.log10(max_count) + 1)
        divider_str = '+' + ('-' * (phone_len + 1))
        divider_str += ('+' + ('-' * (max_count_len + 9))) * 4
        divider_str += '+'
        format_str = '|{{:<{}}}|'.format(phone_len + 1)
        format_str += 4 * '{{:>{}}}({{:05.2f}}%)|'.format(max_count_len + 1)
        print(
            '|{2:<{0}}|{3:>{1}}(%)|{4:>{1}}(%)|{5:>{1}}(%)|{6:>{1}}(%)|'
            ''.format(
                phone_len + 1, max_count_len + 6, 'phone', 'inserts',
                'deletes', 'subs', 'errs',
            ),
            file=out_file,
        )
        print(divider_str, file=out_file)
        print(divider_str, file=out_file)
        for phone in phones:
            i, d, s = inserts[phone], deletes[phone], subs[phone]
            t = totals[phone]
            print(
                format_str.format(
                    phone,
                    i, i / t * 100,
                    d, d / t * 100,
                    s, s / t * 100,
                    i + d + s, (i + d + s) / t * 100,
                ),
                file=out_file
            )
            print(divider_str, file=out_file)
    return 0


def _train_cnn_ctc_parse_args(args, logger):
    parser = KaldiParser(
        description=train_cnn_ctc.__doc__,
        add_verbose=True,
        logger=logger,
    )
    # any boolean configuration should be kaldi-like
    # (e.g. --strict=true)
    parser.register('type', bool, kaldi_bool_arg_type)
    parser.add_argument(
        'data_rspecifier', type='kaldi_rspecifier',
        help='rspecifier to read in audio features used for training')
    parser.add_argument(
        'labels_rspecifier', type='kaldi_rspecifier',
        help='rspecifier to read in labels used for training')
    parser.add_argument(
        'label_to_id_map_path',
        help='Where the file that converts labels to ids is stored.')
    parser.add_argument(
        'val_data_rspecifier', nargs='?', default=None,
        type='kaldi_rspecifier',
        help='Validation data rspecifier. If set, '
             '"val_labels_rspecifier" must be set too')
    parser.add_argument(
        'val_labels_rspecifier', nargs='?', default=None,
        type='kaldi_rspecifier',
        help='Validation labels rspecifier')
    # we add all three sets of arguments as we allow them in the same
    # config file - we just ignore what we don't want
    ModelConfig().add_arguments_to_parser(parser)
    TrainConfig().add_arguments_to_parser(parser)
    DecodeConfig().add_arguments_to_parser(parser)
    options = parser.parse_args(args)
    return options


# adapted from python 3.6 code base
class redirect_stdout_to_stderr(object):
    '''Redirect sys.stdout to sys.stderr for length of context manager'''

    def __init__(self):
        self._old_targets = []

    def __enter__(self):
        self._old_targets.append(sys.stdout)
        new_target = sys.stderr
        sys.stdout = new_target
        return new_target

    def __exit__(self, exctype, excinst, exctb):
        sys.stdout = self._old_targets.pop()


@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def train_cnn_ctc(args=None):
    '''Train CNN w/ CTC decoding using kaldi data tables'''
    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(sys.argv[0])
    options = _train_cnn_ctc_parse_args(args, logger)
    logger.log(9, 'Parsed options')
    label2id_map = dict()
    with open(options.label_to_id_map_path) as file_obj:
        for line in file_obj:
            label, idee = line.strip().split()
            idee = int(idee)
            if idee < 0:
                logger.error('All label ids must be nonnegative')
                return 1
            label2id_map[label] = idee
    if (len(label2id_map) + 1) != options.num_labels:
        logger.error(
            'Expected {} labels in label_to_id_map, got {}'.format(
                options.num_labels - 1, len(label2id_map)))
        return 1
    for idee in range(options.num_labels - 1):
        if idee not in label2id_map.values():
            raise ValueError('label to id map missing id: {}'.format(idee))
    logger.log(9, 'Loaded label to id map')
    model_config = ModelConfig(**vars(options))
    train_config = TrainConfig(**vars(options))
    train_data = TrainData(
        (options.data_rspecifier, 'bm', {'cache': train_config.cache}),
        (options.labels_rspecifier, 'tv', {'cache': train_config.cache}),
        label2id_map,
        batch_size=train_config.batch_size,
    )
    if options.val_data_rspecifier or options.val_labels_rspecifier:
        if None in (
                options.val_data_rspecifier, options.val_labels_rspecifier):
            logger.error(
                "Both 'val_data_rspecifier' and 'val_labels_rspecifier' must "
                "be specified, or neither")
            return 1
        val_data = ValidationData(
            options.val_data_rspecifier,
            options.val_labels_rspecifier,
            label2id_map,
            batch_size=train_config.batch_size,
        )
    else:
        val_data = None
    logger.log(9, 'Set up training/validation data generators')
    with redirect_stdout_to_stderr():
        logger.log(9, 'Creating model')
        from pydrobert.mol.model import ConvCTC
        model = ConvCTC(model_config)
        logger.log(9, 'Beginning training')
        model.fit_generator(train_config, train_data, val_data=val_data)
        logger.log(9, 'Finished training')


def _decode_cnn_ctc_parse_args(args, logger):
    parser = KaldiParser(
        description=decode_cnn_ctc.__doc__,
        add_verbose=True,
        logger=logger,
    )
    # any boolean configuration should be kaldi-like
    # (e.g. --strict=true)
    parser.register('type', bool, kaldi_bool_arg_type)
    parser.add_argument(
        'data_rspecifier', type='kaldi_rspecifier',
        help='rspecifier to read in audio features to decode')
    parser.add_argument(
        'output_wspecifier', type='kaldi_wspecifier',
        help='wspecifier to write decoded sequences to')
    parser.add_argument(
        'label_to_id_map_path',
        help='Where the file that converts labels to ids is stored.')
    # we add all three sets of arguments as we allow them in the same
    # config file - we just ignore what we don't want
    ModelConfig().add_arguments_to_parser(parser)
    TrainConfig().add_arguments_to_parser(parser)
    DecodeConfig().add_arguments_to_parser(parser)
    options = parser.parse_args(args)
    return options


@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def decode_cnn_ctc(args=None):
    '''Decode CNN w/ CTC using kaldi data tables'''
    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(sys.argv[0])
    options = _decode_cnn_ctc_parse_args(args, logger)
    logger.log(9, 'Parsed options')
    id2label_map = dict()
    with open(options.label_to_id_map_path) as file_obj:
        for line in file_obj:
            label, idee = line.strip().split()
            idee = int(idee)
            if idee < 0:
                logger.error('All label ids must be nonnegative')
                return 1
            id2label_map[idee] = label
    if (len(id2label_map) + 1) != options.num_labels:
        logger.error(
            'Expected {} labels in id to label map, got {}'.format(
                options.num_labels - 1, len(id2label_map)))
        return 1
    for idee in range(options.num_labels - 1):
        if idee not in id2label_map.keys():
            logger.error('label to id map missing id: {}'.format(idee))
            return 1
    logger.log(9, 'Loaded label to id map')
    model_config = ModelConfig(**vars(options))
    decode_config = DecodeConfig(**vars(options))
    decode_data = DecodeData(
        options.data_rspecifier,
        batch_size=decode_config.batch_size,
    )
    total_batches = len(decode_data)
    labels_out = io_open(options.output_wspecifier, 'tv', mode='w')
    logger.log(9, 'Set up eval data and opened label output file')
    with redirect_stdout_to_stderr():
        logger.log(9, 'Creating model')
        from pydrobert.mol.model import ConvCTC
        model = ConvCTC(model_config)
        logger.log(9, 'Beginning decoding')
        batches_decoded = 0
        logger.log(9, '000/{:03d} batches decoded'.format(total_batches))
        for label_batch in model.decode_generator(decode_config, decode_data):
            for key, label_ids in label_batch:
                labels_out.write(key, tuple(
                    id2label_map[idee] for idee in label_ids))
        batches_decoded += 1
        if batches_decoded % max(1, total_batches // 10) == 0:
            logger.log(9, '{:03d}/{:03d} batches decoded'.format(
                batches_decoded, total_batches))
    logger.info('Done decoding')


def _normalize_feat_lens_parse_args(args, logger):
    parser = KaldiParser(
        description=normalize_feat_lens.__doc__,
        add_verbose=True,
        logger=logger,
    )
    parser.add_argument(
        'feats_in_rspecifier', type='kaldi_rspecifier',
        help='The features to be normalized (basic matrix table)',
    )
    parser.add_argument(
        'len_in_rspecifier', type='kaldi_rspecifier',
        help='The reference lengths (int32 table)',
    )
    parser.add_argument(
        'feats_out_wspecifier', type='kaldi_wspecifier',
        help='The output features',
    )
    parser.add_argument(
        '--tolerance', type=int, default=float('inf'),
        help='''\
How many frames deviation from reference to tolerate before error. The default
is to be infinitely tolerant (a feat I'm sure we all desire)
''')
    parser.add_argument(
        '--strict', type='kaldi_bool', default=False,
        help='''\
Whether missing keys in either input features or reference raise an error
(true) or are skipped (false)
''')
    parser.add_argument(
        '--pad-mode', default='edge',
        choices=('zero', 'constant', 'edge', 'symmetric', 'mean'),
        help='''\
If frames are being added to the features, specify how they should be added.
zero=zero pad, edge=pad with rightmost frame, symmetric=pad with reverse of
frame edges, mean=pad with mean feature values
''')
    return parser.parse_args(args)


@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def normalize_feat_lens(args=None):
    """Ensure feature matrices match some reference length

    Incoming features are either clipped or padded to match the reference, if
    they are within tolerance (errors if not).

    Input features and reference lengths should both be sorted
    """
    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(sys.argv[0])
    options = _normalize_feat_lens_parse_args(args, logger)
    if options.pad_mode == 'zero':
        options.pad_mode = 'constant'
    logger.info('Parsed options')
    feats_in = io_open(options.feats_in_rspecifier, 'bm', mode='r')
    len_in = io_open(options.len_in_rspecifier, 'i', mode='r')
    feats_out = io_open(options.feats_out_wspecifier, 'bm', mode='w')
    logger.info('Opened readers/writers')
    total_utts = 0
    processed_utts = 0

    def _err_on_utt_id(utt_id, missing_rxspecifier):
        msg = "Utterance '{}' absent in '{}'".format(
            utt_id, missing_rxspecifier)
        if options.strict:
            logger.error(msg)
            return 1
        else:
            logger.warning(msg)
            return 0
    while not feats_in.done() and not len_in.done():
        total_utts += 1
        if feats_in.key() > len_in.key():
            if _err_on_utt_id(len_in.key(), options.feats_in_rspecifier):
                return 1
            len_in.move()
        elif len_in.key() > feats_in.key():
            if _err_on_utt_id(feats_in.key(), options.len_in_rspecifier):
                return 1
            feats_in.move()
        else:
            utt_id = feats_in.key()
            exp_feat_len = len_in.value()
            feats = feats_in.value()
            act_feat_len = len(feats)
            logger.debug('{} exp len: {} act len: {}'.format(
                utt_id, exp_feat_len, act_feat_len))
            if act_feat_len < exp_feat_len:
                if act_feat_len < exp_feat_len - options.tolerance:
                    logger.error(
                        '{} has feature length {}, which is below the '
                        'tolerance ({}) of the expected length {}'.format(
                            utt_id, act_feat_len, options.tolerance,
                            exp_feat_len))
                    return 1
                feats = np.pad(
                    feats,
                    ((0, exp_feat_len - act_feat_len), (0, 0)),
                    options.pad_mode,
                )
                logger.info('Padded {} from length {} to length {}'.format(
                    utt_id, act_feat_len, exp_feat_len))
            elif act_feat_len > exp_feat_len:
                if act_feat_len > exp_feat_len + options.tolerance:
                    logger.error(
                        '{} has feature length {}, which is above the '
                        'tolerance ({}) of the expected length {}'.format(
                            utt_id, act_feat_len, options.tolerance,
                            exp_feat_len))
                    return 1
                feats = feats[:exp_feat_len, :]
                logger.info('Truncated {} from length {} to length {}'.format(
                    utt_id, act_feat_len, exp_feat_len))
            feats_out.write(utt_id, feats)
            processed_utts += 1
            feats_in.move()
            len_in.move()
    while not feats_in.done():
        if _err_on_utt_id(feats_in.key(), options.len_in_rspecifier):
            return 1
        total_utts += 1
        feats_in.move()
    while not len_in.done():
        if _err_on_utt_id(len_in.key(), options.feats_in_rspecifier):
            return 1
        total_utts += 1
        len_in.move()
    logger.info(
        'Processed {}/{} utterances'.format(processed_utts, total_utts))
    return 0


def _find_best_model_from_log_parse_args(args, logger):
    parser = KaldiParser(
        description=find_best_model_from_log.__doc__,
        add_verbose=True,
        logger=logger,
    )
    parser.add_argument('csv_path', help='The training log to check')
    parser.add_argument('--csv-delimiter', default=',')
    parser.add_argument(
        '--training-stage', default=None,
        help='If set, limits the search only to this training stage')
    parser.add_argument(
        '--monitored', default=None,
        help='''\
The value monitored to be the "best." By default, uses val_acc if available,
then val_loss, then acc, then loss''')
    parser.add_argument(
        '--mode', choices=['min', 'max', 'auto'], default='auto',
        help='What direction of a changed in the monitored value is good')
    return parser.parse_args(args)


@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def find_best_model_from_log(args=None):
    '''Find the best model from training log and return its path on stdout

    In a given training stage, the 'best' model is the one that minimizes or
    maximizes the target quantity (dependent on 'mode' and 'monitored').
    '''
    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(sys.argv[0])
    options = _find_best_model_from_log_parse_args(args, logger)
    logger.log(9, 'Parsed options')
    csv_file = open(options.csv_path, mode='r')
    csv_reader = DictReader(csv_file, delimiter=options.csv_delimiter)
    fields = csv_reader.fieldnames
    if 'model_path' not in fields:
        logger.error('"model_path" must be a field in the csv')
        return 1
    if options.monitored is None:
        if 'val_acc' in fields:
            options.monitored = 'val_acc'
        elif 'val_loss' in fields:
            options.monitored = 'val_loss'
        elif 'acc' in fields:
            options.monitored = 'acc'
        elif 'loss' in fields:
            options.monitored = 'loss'
        else:
            logger.error(
                'Unable to find suitable value to monitor. Set --monitored '
                'manually')
            return 1
        logger.log(
            9, 'Using "{}" as monitored value'.format(options.monitored))
    elif options.monitored not in fields:
        logger.error('Monitored value "{}" not in CSV'.format(
            options.monitored))
        return 1
    if options.mode == 'min' or (
            options.mode == 'auto' and 'acc' not in options.monitored):
        monitor_op = np.less
        best = np.Inf
    else:  # max
        monitor_op = np.greater
        best = -np.Inf
    if options.training_stage and 'training_stage' not in fields:
        logger.error(
            '--training-stage specified but training_stage not in csv')
        return 1
    logger.log(9, 'Looking through CSV')
    best_path = None
    for row in csv_reader:
        if options.training_stage and (
                row['training_stage'] != options.training_stage):
            continue
        if monitor_op(row[options.monitored], best):
            best = row[options.monitored]
            best_path = row['model_path']
    logger.log(9, 'Looked through CSV')
    if best_path is None:
        logger.error('Could not find any model')
        return 1
    else:
        print(best_path)
        return 0
