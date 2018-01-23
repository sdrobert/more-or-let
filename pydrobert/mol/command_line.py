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

from pydrobert.kaldi.io import open as io_open
from pydrobert.kaldi.io.argparse import KaldiParser
from pydrobert.kaldi.io.argparse import kaldi_bool_arg_type
from pydrobert.kaldi.logging import kaldi_logger_decorator
from pydrobert.kaldi.logging import kaldi_vlog_level_cmd_decorator
from pydrobert.kaldi.logging import register_logger_for_kaldi
from pydrobert.mol.config import DecodeConfig
from pydrobert.mol.config import ModelConfig
from pydrobert.mol.config import TrainConfig
from pydrobert.mol.corpus import DecodeData
from pydrobert.mol.corpus import TrainData
from pydrobert.mol.corpus import ValidationData
from pydrobert.mol.util import CMVNCalculator
from pydrobert.mol.util import calculate_deltas

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

__all__ = [
    'train_cnn_ctc',
    'decode_cnn_ctc',
    'find_best_model_from_log',
    'alt_compute_cmvn_stats',
    'alt_apply_cmvn',
    'alt_add_deltas'
]


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
        delta_order=model_config.delta_order,
        cmvn_rxfilename=model_config.cmvn_rxfilename,
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
            delta_order=model_config.delta_order,
            cmvn_rxfilename=model_config.cmvn_rxfilename,
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
        delta_order=model_config.delta_order,
        cmvn_rxfilename=model_config.cmvn_rxfilename,
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
            if decode_data.batch_size:
                for key, label_ids in label_batch:
                    labels_out.write(key, tuple(
                        id2label_map[idee] for idee in label_ids))
            else:
                labels_out.write(label_batch[0], tuple(label_batch[1]))
            batches_decoded += 1
            if batches_decoded % max(1, total_batches // 10) == 0:
                logger.log(9, '{:03d}/{:03d} batches decoded'.format(
                    batches_decoded, total_batches))
    logger.info('Done decoding')


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
        current = float(row[options.monitored])
        if monitor_op(current, best):
            best = current
            best_path = row['model_path']
    logger.log(9, 'Looked through CSV')
    if best_path is None:
        logger.error('Could not find any model')
        return 1
    else:
        print(best_path)
        return 0


def _alt_compute_cmvn_stats_parse_args(args, logger):
    parser = KaldiParser(
        description=alt_compute_cmvn_stats.__doc__,
        add_verbose=True,
        logger=logger,
    )
    parser.add_argument(
        'feats_in', type='kaldi_rspecifier',
        help='Features to compute over'
    )
    parser.add_argument(
        'cmvn_stats_out', type='kaldi_wxfilename',
        help='Where to store global CMVN stats'
    )
    return parser.parse_args(args)


@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def alt_compute_cmvn_stats(args=None):
    '''Python-based code for CMVN statistics computation

    Used for debugging
    '''
    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(sys.argv[0])
    options = _alt_compute_cmvn_stats_parse_args(args, logger)
    cmvn = CMVNCalculator()
    feat_table = io_open(options.feats_in, 'bm')
    num_utts = 0
    for feats in feat_table:
        cmvn.accumulate(feats)
        num_utts += 1
    logger.info('Accumulated stats for {} utterances'.format(num_utts))
    cmvn.save(options.cmvn_stats_out)
    logger.info('Wrote stats to {}'.format(options.cmvn_stats_out))


def _alt_apply_cmvn_parse_args(args, logger):
    parser = KaldiParser(
        description=alt_apply_cmvn.__doc__,
        add_verbose=True,
        logger=logger,
    )
    parser.add_argument(
        'cmvn_stats_in', type='kaldi_rxfilename',
        help='Where global CMVN stats are stored',
    )
    parser.add_argument(
        'feats_in', type='kaldi_rspecifier',
        help='Features to apply to',
    )
    parser.add_argument(
        'feats_out', type='kaldi_wspecifier',
        help='Where to write normalized features',
    )
    return parser.parse_args(args)


@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def alt_apply_cmvn(args=None):
    '''Python-based code for CMVN application

    Used for debugging
    '''
    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(sys.argv[0])
    options = _alt_apply_cmvn_parse_args(args, logger)
    cmvn = CMVNCalculator(options.cmvn_stats_in)
    feats_in = io_open(options.feats_in, 'bm')
    feats_out = io_open(options.feats_out, 'bm', mode='w')
    num_utts = 0
    for utt_id, feats in feats_in.items():
        feats = cmvn.apply(feats, in_place=True)
        feats_out.write(utt_id, feats)
        num_utts += 1
    logger.info('Applied CMVN to {} utterances'.format(num_utts))


def _alt_add_deltas_parse_args(args, logger):
    parser = KaldiParser(
        description=alt_add_deltas.__doc__,
        logger=logger,
    )
    parser.add_argument(
        'feats_in', type='kaldi_rspecifier',
        help='Features to apply to',
    )
    parser.add_argument(
        'feats_out', type='kaldi_wspecifier',
        help='Where to write features to'
    )
    parser.add_argument(
        '--delta-order', type=int, default=2,
        help='How many deltas to add'
    )
    return parser.parse_args(args)


@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def alt_add_deltas(args=None):
    '''Python-based code for adding deltas

    Used for debugging
    '''
    logger = logging.getLogger(sys.argv[0])
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(sys.argv[0])
    options = _alt_add_deltas_parse_args(args, logger)
    feats_in = io_open(options.feats_in, 'bm')
    feats_out = io_open(options.feats_out, 'bm', mode='w')
    num_utts = 0
    for utt_id, feats in feats_in.items():
        feats = calculate_deltas(feats, options.delta_order)
        feats_out.write(utt_id, feats)
        num_utts += 1
    logger.info('Added {} deltas to {} utterances'.format(
        options.delta_order, num_utts))
