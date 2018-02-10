'''Common script-like methods'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pydrobert.kaldi.io import open as kaldi_open
from pydrobert.kaldi.io.enums import KaldiDataType

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"


def calculate_deltas(data, num_deltas, axis=0, target_axis=-1, concatenate=True):
    '''Calculate deltas for arrays

    Deltas are simply weighted rolling averages; double deltas are the
    rolling averages of rolling averages. This can be done an arbitrary
    number of times. Because most signals are non-zero in silence, the
    signal is edge-padded before convolution.

    Parameters
    ----------
    data : array-like
        At least one dimensional
    num_deltas : int
        A nonnegative integer specifying the number of delta calculations
    axis : int
        The axis of `data` to be calculated over (i.e. convolved)
    target_axis : int
        The location where the new axis, for deltas, will be inserted
    concatenate : bool
        Whether to concatenate the deltas to the end of `target_axis` (`True`),
        or create a new axis in this location (`False`)

    Returns
    -------
    array-like
    '''
    max_filt_width = 4 * num_deltas + 1
    pad_widths = [(0, 0)] * len(data.shape)
    pad_widths[axis] = ((max_filt_width - 1) // 2, (max_filt_width - 1) // 2)
    slices = [slice(None)]
    slices[axis] = slice((max_filt_width - 1) // 2, -(max_filt_width - 1) // 2)
    delta_data_list = [data]
    padded_data = np.pad(data, pad_widths, 'edge')
    delta_filt = np.asarray((.2, .1, 0., -.1, -.2))
    cur_filt = np.ones(1)
    for _ in range(num_deltas):
        cur_filt = np.convolve(cur_filt, delta_filt, 'full')
        delta_data_list.append(np.apply_along_axis(
            np.convolve, axis, padded_data, cur_filt, 'same')[slices])
    if concatenate:
        return np.concatenate(delta_data_list, target_axis)
    else:
        return np.stack(delta_data_list, target_axis)


class CMVNCalculator(object):
    '''Perform unit normalization on each row of features

    Instances of this class accumulate means and sums of squares over
    features. Later features can then be transformed according to those
    statistics to something close to zero mean and unit variance.

    The implementation is based off Kaldi's, but doesn't do casting to
    double precision in intermediate values

    Parameters
    ----------
    rxfilename : str, optional
        If provided, sufficient statistics will be loaded from this
        extended file
    along_axis : int
        What axis to compute statistics over

    Attributes
    ----------
    stats : array-like
        Stored stats matrix
    along_axis : int
        What axis to compute statistics over
    '''

    def __init__(self, rxfilename=None, along_axis=0):
        self.along_axis = along_axis
        if KaldiDataType.BaseMatrix.is_double:
            self.stats_dtype = np.float64
        else:
            self.stats_dtype = np.float32
        if rxfilename:
            with kaldi_open(rxfilename) as stats_file:
                self.stats = stats_file.read('bm')
        else:
            self.stats = None

    def accumulate(self, feats):
        '''Add feature matrix statistics to counts in key

        Parameters
        ----------
        feats : array-like
            A 2D array to accumulate over
        '''
        if self.stats is None:
            self.stats = np.zeros(
                (2, feats.shape[1 - self.along_axis] + 1),
                dtype=self.stats_dtype,
            )
        self.stats[0, -1] += feats.shape[self.along_axis]
        self.stats[0, :-1] += feats.sum(self.along_axis)
        self.stats[1, :-1] += (feats ** 2).sum(self.along_axis)

    def apply(self, feats, in_place=False):
        '''Apply CMVN to features

        Parameters
        ----------
        feats : array-like
            2D array to transform
        in_place : bool
            Whether to apply the transformation in place (if possible)
            or copy the feature matrix

        Raises
        ------
        KeyError
        '''
        if feats.ndim != 2:
            raise ValueError(
                'Expected 2-dimensional feature matrix, got {}'.format(
                    feats.ndim))
        count = self.stats[0, -1]
        means = self.stats[0,:-1] / count
        varss = np.maximum(self.stats[1, :-1] / count - np.square(means), 1e-5)
        scales = 1 / np.sqrt(varss)
        offsets = -means * scales
        if not in_place:
            feats = feats.copy()
        if self.along_axis:
            scales = scales[:, np.newaxis]
            offsets = offsets[:, np.newaxis]
        feats *= scales
        return feats

    def save(self, wxfilename):
        '''Save statistics to extended file name'''
        with kaldi_open(wxfilename, mode='w') as stats_file:
            stats_file.write(self.stats, 'bm')
