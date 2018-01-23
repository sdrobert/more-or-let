'''Corpus classes and utilities'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import str as text

import numpy as np

from pydrobert.kaldi.io.corpus import SequentialData
from pydrobert.kaldi.io.corpus import ShuffledData
from pydrobert.mol.util import CMVNCalculator
from pydrobert.mol.util import calculate_deltas

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"


__all__ = [
    'TrainData',
    'ValidationData',
    'DecodeData',
]


def _loss_data_wrapper(cls):
    '''Wrap either ShuffledData or SequentialData for loss calcs'''
    class _Wrapper(cls):
        '''Data for training/validation

        From two tables: feats (matrices) and labels (token vectors)

        Yields minibatches of:
            ([feats, labels, feat_sizes, label_sizes], [dummy])

        where:
        - feats is a float array of shape (batch_size, max_n_frames,
          n_feats, 1) containing left-aligned edge-padded features
        - labels is an int32 array of shape (batch_size, max_n_labels)
          containing left-aligned padded label ids (acquired via lookup
          of label2id_map). The padding value is unspecified (since it
          is never used).
        - feat_sizes is an int32 array of shape (batch_size, 1)
          indicating the original n_frames of each sample in the batch
        - label_sizes is an int32 array of shape (batch_size, 1)
          indicating the original n_labels of each sample in the batch
        - dummy is a float array (of same type as feats) of shape
          (batch_size,) with unspecified values
        '''

        def __init__(
                self, feat_table, label_table, label2id_map,
                delta_order=0, cmvn_rxfilename=None, **kwargs):
            for invalid_kwarg in ('axis_lengths', 'batch_pad_mode', 'add_key'):
                if invalid_kwarg in kwargs:
                    raise TypeError(
                        'Invalid keyword argument "{}"'.format(invalid_kwarg))
            self.delta_order = delta_order
            if cmvn_rxfilename is not None:
                self.cmvn = CMVNCalculator(cmvn_rxfilename)
            else:
                self.cmvn = None
            self.label2id_map = label2id_map
            self.blank_label = max(label2id_map.values()) + 1
            if isinstance(label_table, str) or isinstance(label_table, text):
                label_table = (label_table, 'tv')
            super(_Wrapper, self).__init__(
                feat_table, label_table,
                axis_lengths=((0, 0), (1, 0)),
                batch_pad_mode='edge',
                add_key=False,
                **kwargs
            )

        def batch_generator(self, repeat=False):
            for feats, labels, feat_sizes, label_sizes in super(
                    _Wrapper, self).batch_generator(repeat=repeat):
                if not self.batch_size:
                    feats = np.expand_dims(feats, 0)
                    labels = [labels]
                    feat_sizes = [feat_sizes]
                    label_sizes = [label_sizes]
                if self.cmvn or self.delta_order:
                    new_shape = list(feats.shape)
                    num_feats = new_shape[2]
                    new_shape[2] *= (1 + self.delta_order)
                    new_feats = np.empty(new_shape, dtype=feats.dtype)
                    for samp_idx in range(len(feats)):
                        if self.cmvn:
                            new_feats[samp_idx, :, :num_feats] = (
                                self.cmvn.apply(feats[samp_idx], in_place=True)
                            )
                        else:
                            new_feats[samp_idx, :, :num_feats] = (
                                feats[samp_idx]
                            )
                        if self.delta_order:
                            new_feats[samp_idx] = calculate_deltas(
                                new_feats[samp_idx, :, :num_feats],
                                self.delta_order
                            )
                    feats = new_feats
                feats = np.expand_dims(feats, -1)
                max_label_sequence_len = max(len(seq) for seq in labels)
                label_batch = np.ones(
                    (len(labels), max_label_sequence_len),
                    dtype=np.int32
                ) * self.blank_label
                for samp_idx, seq in enumerate(labels):
                    label_batch[samp_idx, :len(seq)] = tuple(
                        self.label2id_map[lab] for lab in seq)
                feat_sizes = np.expand_dims(
                    np.array(feat_sizes, dtype=np.int32, copy=False), -1)
                label_sizes = np.expand_dims(
                    np.array(label_sizes, dtype=np.int32, copy=False), -1)
                dummy = np.zeros(feat_sizes.shape, dtype=feats.dtype)
                yield [feats, label_batch, feat_sizes, label_sizes], [dummy]

    return _Wrapper


TrainData = _loss_data_wrapper(ShuffledData)


ValidationData = _loss_data_wrapper(SequentialData)


class DecodeData(SequentialData):
    '''Data for decoding

    From a single table of feats (matrices), yields minibatches of:
        (utt_ids, feats, feat_sizes)

    where:
        - utt_ids is a list of strings of length batch_size indicating
          what utterances samples in feats are associated with
        - feats is a float array of shape (batch_size, max_n_frames,
          n_feats, 1) containing left-aligned edge-padded features
        - feat_sizes is an int32 array of shape (batch_size, 1)
          indicating the original n_frames of each sample in the batch
    '''

    def __init__(
            self, feat_table, delta_order=0, cmvn_rxfilename=None, **kwargs):
        for invalid_kwarg in ('axis_lengths', 'batch_pad_mode', 'add_key'):
            if invalid_kwarg in kwargs:
                raise TypeError(
                    'Invalid keyword argument "{}"'.format(invalid_kwarg))
        self.delta_order = delta_order
        if cmvn_rxfilename is not None:
            self.cmvn = CMVNCalculator(cmvn_rxfilename)
        else:
            self.cmvn = None
        super(DecodeData, self).__init__(
            feat_table,
            axis_lengths=(0, 0),
            add_key=True,
            batch_pad_mode='edge',
            **kwargs
        )

    def batch_generator(self, repeat=False):
        for keys, feats, feat_sizes in super(DecodeData, self).batch_generator(
                repeat=repeat):
            if not self.batch_size:
                keys = [keys]
                feats = np.expand_dims(feats, 0)
                feat_sizes = [feat_sizes]
            if self.cmvn or self.delta_order:
                new_shape = list(feats.shape)
                num_feats = new_shape[2]
                new_shape[2] *= (1 + self.delta_order)
                new_feats = np.empty(new_shape, dtype=feats.dtype)
                for samp_idx in range(len(feats)):
                    if self.cmvn:
                        new_feats[samp_idx, :, :num_feats] = self.cmvn.apply(
                            feats[samp_idx], in_place=True)
                    else:
                        new_feats[samp_idx, :, :num_feats] = feats[samp_idx]
                    if self.delta_order:
                        new_feats[samp_idx] = calculate_deltas(
                            new_feats[samp_idx, :, :num_feats],
                            self.delta_order
                        )
                feats = new_feats
            feats = np.expand_dims(feats, -1)
            feat_sizes = np.expand_dims(
                np.array(feat_sizes, dtype=np.int32), -1)
            yield keys, feats, feat_sizes
