'''Tensor ops and models related to Connectionist Temporal Classification

All CTC classes accept input via generators in one of the following forms:

1. Numpy arrays of shape `(time, bank_size)` representing individual audio
   samples. This is used for decoding
2. Tuples of `(audio_sample, label_seq)`, where `label_seq` is an
   array-like of shape `(num_labels,)` whose values are the label sequence.
   This is used for training

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import listdir
from os import makedirs
from os.path import isdir
from os.path import join
from sys import stderr

import keras.backend as K
import numpy as np

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ProgbarLogger
from keras.callbacks import TensorBoard
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers.merge import Maximum
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.regularizers import l2
from pydrobert.signal.post import CMVN
from pydrobert.signal.post import Deltas

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

class ConvCTC(object):
    '''Fully convolutional CTC

    Architecture is based off [1]_ entirely.

    Input are expected to be of the form described in module docstring

    Parameters
    ----------
    input_shape : tuple
        Tuple of `(max_time_steps, bank_size)`. `bank_size` needs to be
        set, but not `max_time_steps`.
    num_labels : int
        Number of unique labels, including the blank
    num_deltas : int, optional
        The number of deltas to calculate on the audio. Deltas will be
        appended on the second axis
    weight_dir : str, optional
        Path to where weights are/will be stored. Default is to store no
        weights
    fine_tuning : bool, optional
        Whether the model has entered the 'fine-tuning' stage of
        training
    tensorboard_dir : str, optional
        If set to a valid path and the Keras backend is Tensorflow,
        writes Tensorboard logs to that path
    cmvn_path : str, optional
        If set, CMVN stats will be drawn from this file and applied to
        every utterance
    concatenate : bool
        Whether deltas are concatenated along the frequency axis
        (`True`), or given their own channels (`False`)
    double_weights : bool
        If enabled, feature maps and hidden units are doubled to combat
        the ambiguity in the weight counts of [1]_

    Attributes
    ----------
    input_shape : tuple
    num_labels : int
    weight_dir : str or None
    fine_tuning : boolstopping
    tensorboard_dir : str or None

    .. [1] Zhang, Y et al. "Towards End-to-End Speech Recognition with Deep
       Convolutional Neural Networks" https://arxiv.org/abs/1701.02720
    '''

    def __init__(
            self, input_shape, num_labels, num_deltas=0,
            weight_dir=None, fine_tuning=False, tensorboard_dir=None,
            cmvn_path=None, concatenate=True, double_weights=False):
        if not input_shape[1]:
            raise ValueError('bank size must be fixed')
        if K.image_dim_ordering() != 'tf':
            # not sure if I'm right, but I think the TimeDistributed
            # wrapper will always take axis 1, which could be the
            # channel axis in Theano
            raise ValueError('dimensions must be tensorflow-ordered')
        if weight_dir is not None and not isdir(weight_dir):
            makedirs(weight_dir)
        if tensorboard_dir is not None:
            if K.backend() != 'tensorflow':
                print(
                    'Ignoring tensorboard_dir setting. Backend is not '
                    'tensorflow',
                    file=stderr
                )
                tensorboard_dir = None
            elif not isdir(tensorboard_dir):
                makedirs(tensorboard_dir)
        self._tensorboard_dir = tensorboard_dir
        self._weight_dir = weight_dir
        self._num_labels = num_labels
        self._input_shape = input_shape
        self._fine_tuning = fine_tuning
        if num_deltas:
            self._deltas = Deltas(num_deltas, concatenate=concatenate)
        else:
            self._deltas = None
        self._audio_input = None
        self._audio_size_input = None
        self._label_input = None
        self._label_size_input = None
        self._activation_layer = None
        self._acoustic_model = None
        self._double_weights = double_weights
        if cmvn_path:
            self._cmvn = CMVN(cmvn_path, dtype='bm')
        else:
            self._cmvn = CMVN()
        # constants or initial settings based on paper
        self._filt_size = (5, 3) # time first, unlike paper
        self._pool_size = (1, 3)
        self._dropout_p = 0.3
        # I asked the first author about this. To keep the number of
        # parameters constant for maxout, she halved the values she
        # reported in the paper
        self._initial_filts = 128 // (1 if double_weights else 2)
        self._dense_size = 1024 // (1 if double_weights else 2)
        self._layer_kwargs = {
            'activation': 'linear',
            'kernel_initializer' : 'uniform',
        }
        if self._fine_tuning:
            self._layer_kwargs['kernel_regularizer'] = l2(l=1e-5)
        self._construct_acoustic_model()
        self._past_epochs = 0
        self._acoustic_model.summary()
        super(ConvCTC, self).__init__()

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def weight_dir(self):
        return self._weight_dir

    @property
    def fine_tuning(self):
        return self._fine_tuning

    @property
    def tensorboard_dir(self):
        return self._tensorboard_dir

    def fit_generator(
            self, train_generator, train_samples_per_epoch,
            val_generator=None, val_samples_per_epoch=None,
            batch_size=20, max_epochs=1000000, early_stopping=None):
        '''Fit the acoustic model to data from generators

        Parameters
        ----------
        train_generator : generator
            An infinitely cycling generator of audio input and labels to train
        train_samples_per_epoch : int
            How many unique samples are generated
        val_generator : generator, optional
            Validation set generator. If not set, training loss is used to
            keep track of best/early stopping
        val_samples_per_epoch : int, optional
            Must be set if `val_generator` is set
        batch_size : int, optional
            Size of minibatches. Decrease if running out of memory
        max_epochs : int, optional
            The maximum number of epochs to run for. Can be fewer if early
            stopping is enabled. The default is a million epochs (practically
            infinite)
        early_stopping : int or None
            If `None`, early stopping is disabled. Otherwise this value is used
            as the `patience` parameter for early stopping
        '''
        val_monitored = None
        if val_generator or val_samples_per_epoch:
            if None in (val_generator, val_samples_per_epoch):
                raise ValueError(
                    'Either both val_generator and val_samples_per_epoch '
                    'must be set, or neither'
                )
            val_monitored = 'val_loss'
            val_generator = _training_wrapper(
                val_generator,
                batch_size,
                val_samples_per_epoch,
                self._num_labels,
                self._deltas,
                self._cmvn,
            )
        else:
            print(
                'Monitoring training loss instead of validation loss. Not '
                'recommended.', file=stderr)
            val_monitored = 'loss'
        train_generator = _training_wrapper(
            train_generator,
            batch_size,
            train_samples_per_epoch,
            self._num_labels,
            self._deltas,
            self._cmvn,
        )
        callbacks = []
        optimizer = None
        if self._fine_tuning:
            optimizer = SGD(lr=1e-5, clipvalue=1.0)
        else:
            optimizer = Adam(lr=1e-4, clipvalue=1.0)
        if self._weight_dir:
            file_regex = join(
                self._weight_dir,
                'weights.{epoch:03d}.{' + val_monitored + ':07.2f}.hdf5',
            )
            callbacks.append(ModelCheckpoint(
                filepath=file_regex, monitor=val_monitored,
                save_weights_only=True, save_best_only=False,
                period=1,
            ))
        if self._tensorboard_dir:
            callbacks.append(TensorBoard(
                log_dir=self._tensorboard_dir, write_graph=False,
                batch_size=batch_size))
        if early_stopping is not None:
            callbacks.append(EarlyStopping(
                monitor=val_monitored,
                patience=early_stopping,
                mode='min',
                min_delta=.1,
            ))
        self._acoustic_model.compile(
            loss={'ctc_loss' : lambda y_true, y_pred: y_pred},
            optimizer=optimizer,
        )
        self._load_weights()
        self._acoustic_model.fit_generator(
            train_generator,
            (train_samples_per_epoch + batch_size - 1) // batch_size,
            max_epochs,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=(val_samples_per_epoch + batch_size - 1) // batch_size,
            initial_epoch=self._past_epochs,
        )

    def decode_once(self, audio, beam_width=100):
        '''Decode a sample of audio using beam search

        Parameters
        ----------
        audio : array-like
            2D array of shape `(time, freq)`
        beam_width : int

        Returns
        -------
        array-like
            Either a 1D array of `(num_labels,)`
        '''
        self._load_weights(for_decoding=True)
        decoder = self._construct_decoder(beam_width)
        if self._cmvn:
            audio = self._cmvn.apply(audio, axis=1, in_place=True)
        if self._deltas:
            audio = self._deltas.apply(audio, axis=0, in_place=True)
        length_batch = np.asarray([[audio.shape[0]]], dtype=np.int32)
        ret_labels = decoder(
            [audio[np.newaxis, :, :, np.newaxis], length_batch, 0]
        )[0][0]
        return tuple(int(idee) for idee in ret_labels if idee != -1)

    def decode_generator(self, generator, num_samples, beam_width=100):
        '''Decode audio samples from a generator using beam search

        Parameters
        ----------
        generator : generator
            Generators can follow the same style as in `fit_generator` (the
            label tensors are discarded), or can contain the audio
            samples/batches by themselves
        num_samples : int, optional
            Stop after this number of samples. The generator is expected to
            produce infinitely, so this will be after num_samples
        beam_width : int

        Yields
        ------
        tuple or None
            Sequences of labels, one sample at a time
        '''
        self._load_weights(for_decoding=True)
        decoder = self._construct_decoder(beam_width)
        wrapper = _decoding_wrapper(generator, self._deltas, self._cmvn)
        for _ in range(num_samples):
            cur_batch = next(wrapper)
            ret_labels = decoder(cur_batch + [0])[0][0]
            yield tuple(int(idee) for idee in ret_labels if idee != -1)

    def _construct_acoustic_model(self):
        # construct acoustic model
        # convolutional layer pattern
        def _conv_maxout_layer(last_layer, n_filts, name_prefix, dropout=True):
            conv_a = Conv2D(
                n_filts, self._filt_size,
                padding='same',
                name=name_prefix + '_a', **self._layer_kwargs
            )(last_layer)
            conv_b = Conv2D(
                n_filts, self._filt_size,
                padding='same',
                name=name_prefix + '_b', **self._layer_kwargs
            )(last_layer)
            last = Maximum(name=name_prefix + '_m')([conv_a, conv_b])
            # pre-weights (i.e. post max), as per
            # http://jmlr.org/proceedings/papers/v28/goodfellow13.pdf
            if dropout:
                last = Dropout(self._dropout_p, name=name_prefix + '_d')(last)
            return last
        n_filts = self._initial_filts
        # inputs
        audio_input_shape = [self._input_shape[0], self._input_shape[1], 1]
        if self._deltas:
            if self._deltas.concatenate:
                audio_input_shape[1] *= self._deltas.num_deltas + 1
            else:
                audio_input_shape[2] *= self._deltas.num_deltas + 1
        self._audio_input = Input(
            shape=audio_input_shape, name='audio_in')
        self._audio_size_input = Input(shape=(1,), name='audio_size_in')
        self._label_input = Input(shape=(None,), name='label_in')
        self._label_size_input = Input(shape=(1,), name='label_size_in')
        last_layer = self._audio_input
        # convolutional layers
        last_layer = _conv_maxout_layer(
            last_layer, n_filts, 'conv_1', dropout=False)
        last_layer = MaxPooling2D(
            pool_size=self._pool_size, name='conv_1_p')(last_layer)
        last_layer = Dropout(self._dropout_p, name='conv_1_d')(last_layer)
        for layer_no in range(2, 11):
            if layer_no == 5:
                n_filts *= 2
            last_layer = _conv_maxout_layer(
                last_layer, n_filts, 'conv_{}'.format(layer_no))
        last_layer = Lambda(
            lambda layer: K.max(layer, axis=2),
            output_shape=(
                self._input_shape[0],
                n_filts,
            ),
            name='max_freq_into_channel',
        )(last_layer)
        # dense layers
        for layer_no in range(1, 4):
            name_prefix = 'dense_{}'.format(layer_no)
            dense_a = Dense(
                self._dense_size, name=name_prefix + '_a',
                **self._layer_kwargs
            )
            dense_b = Dense(
                self._dense_size, name=name_prefix + '_b',
                **self._layer_kwargs
            )
            td_a = TimeDistributed(
                dense_a, name=name_prefix + '_td_a')(last_layer)
            td_b = TimeDistributed(
                dense_b, name=name_prefix + '_td_b')(last_layer)
            last_layer = Maximum(name=name_prefix + '_m')([td_a, td_b])
            last_layer = Dropout(
                self._dropout_p, name=name_prefix + '_d'
            )(last_layer)
        activation_dense = Dense(
            self._num_labels, name='dense_activation',
            **self._layer_kwargs
        )
        self._activation_layer = TimeDistributed(
            activation_dense, name='dense_activation_td')(last_layer)
        # we take a page from the image_ocr example and treat the ctc as a
        # lambda layer.
        self._loss_layer = Lambda(
            lambda args: _ctc_loss(*args),
            output_shape=(1,), name='ctc_loss'
        )([
            self._label_input,
            self._activation_layer,
            self._audio_size_input,
            self._label_size_input
        ])
        self._acoustic_model = Model(
            inputs=[
                self._audio_input,
                self._label_input,
                self._audio_size_input,
                self._label_size_input,
            ],
            outputs=[self._loss_layer],
        )

    def _construct_decoder(self, beam_width):
        label_out = _ctc_decode(
            self._activation_layer, self._audio_size_input,
            beam_width=beam_width
        )
        decoder = K.function(
            [self._audio_input, self._audio_size_input, K.learning_phase()],
            label_out,
        )
        return decoder

    def _load_weights(self, for_decoding=False):
        to_load = None
        self._past_epochs = 0
        if self._weight_dir:
            # weight filename format is `weights.epoch.val_loss.hdf5`
            if for_decoding:
                # load the weights with the lowest validation loss
                min_loss = float('inf')
                for name in listdir(self._weight_dir):
                    cur_loss = float(str.join('.', name.split('.')[2:-1]))
                    cur_epoch = int(name.split('.')[1]) + 1
                    if cur_loss < min_loss:
                        to_load = name
                        min_loss = cur_loss
                    self._past_epochs = max(cur_epoch, self._past_epochs)
            else:
                # load the last set of stored weights
                for name in listdir(self._weight_dir):
                    cur_epoch = int(name.split('.')[1]) + 1
                    if cur_epoch > self._past_epochs:
                        self._past_epochs = cur_epoch
                        to_load = name
        if to_load:
            self._acoustic_model.load_weights(join(self._weight_dir, to_load))
        else:
            print('No weights to load!', file=stderr)

def _decoding_wrapper(wrapped, deltas, cmvn):
    # samples are tuples of (audio, audio_len). Return batches of size 1
    while True:
        elem = next(wrapped)
        audio = None
        if isinstance(elem, np.ndarray):
            audio = elem
        else:
            audio = elem[0] # assume tuple w/ index 1 the label
        if cmvn:
            audio = cmvn.apply(audio, axis=1, in_place=True)
        if deltas:
            audio = deltas.apply(audio, axis=0, in_place=True)
        length_batch = np.asarray([[audio.shape[0]]], dtype=np.int32)
        inputs = [audio[np.newaxis, :, :, np.newaxis], length_batch]
        yield inputs
        del audio, length_batch

def _training_wrapper(
        wrapped, max_batch_size, epoch_size, blank_label, deltas, cmvn):
    # samples are tuples of (audio, labels, audio_len, label_len)
    elems = [next(wrapped)]
    bank_size = elems[0][0].shape[1]
    remainder = epoch_size
    while True:
        batch_size = min(remainder, max_batch_size)
        dummy_y_true = np.zeros((batch_size,), dtype=K.floatx())
        while len(elems) < batch_size:
            elems.append(next(wrapped))
        max_time_steps = max(elem[0].shape[0] for elem in elems)
        max_label_length = max(len(elem[1]) for elem in elems)
        batch_shape = [batch_size, max_time_steps, bank_size, 1]
        if deltas:
            if deltas.concatenate:
                batch_shape[2] *= deltas.num_deltas + 1
            else:
                batch_shape[3] *= deltas.num_deltas + 1
        audios = np.empty(batch_shape, dtype=K.floatx())
        # should be ints, but I think they're cast by keras to floats.
        # Might as well not double dip
        label_seqs = np.ones((batch_size, max_label_length), dtype=K.floatx())
        label_seqs *= blank_label
        audio_lengths = np.empty((batch_size, 1), dtype=K.floatx())
        label_lengths = np.empty((batch_size, 1), dtype=K.floatx())
        for samp_idx, (audio, label_seq) in enumerate(elems):
            audio = np.pad(
                audio, ((0, max_time_steps - audio.shape[0]), (0, 0)), 'edge')
            if cmvn:
                audio = cmvn.apply(audio, axis=1, in_place=True)
            if deltas:
                audio = deltas.apply(audio, axis=0, in_place=True)
            audios[samp_idx, :, :, :] = audio.reshape(batch_shape[1:])
            audio_lengths[samp_idx, 0] = audio.shape[0]
            label_seqs[samp_idx, :len(label_seq)] = label_seq
            label_lengths[samp_idx, 0] = len(label_seq)
        inputs = {
            'audio_in' : audios,
            'audio_size_in' : audio_lengths,
            'label_in' : label_seqs,
            'label_size_in' : label_lengths,
        }
        outputs = {
            'ctc_loss' : dummy_y_true,
        }
        elems = []
        remainder -= batch_size
        if not remainder:
            remainder = epoch_size
        yield inputs, outputs
        del audios, label_seqs, audio_lengths, label_lengths, dummy_y_true

def _dft_ctc_loss(y_true, y_pred, input_length, label_length):
    # keras impl assumes softmax then log hasn't been performed yet. In tf.nn,
    # it has
    assert False, "fixme"
    sm_y_pred = K.softmax(y_pred)
    cost = K.ctc_batch_cost(y_true, sm_y_pred, input_length, label_length)
    return cost

def _dft_ctc_decode(y_pred, input_length, beam_width=100):
    assert False, "fixme"
    sm_y_pred = K.softmax(y_pred)
    return K.ctc_decode(
        sm_y_pred, K.flatten(input_length),
        beam_width=beam_width, greedy=False, top_paths=1)[0][0]

def _tf_dft_ctc_decode(y_pred, input_length, beam_width=100):
    import tensorflow as tf
    input_length = tf.to_int32(tf.reshape(input_length, [-1]))
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    (decoded,), _ = tf.nn.ctc_beam_search_decoder(
        inputs=y_pred,
        sequence_length=input_length,
        beam_width=beam_width,
        top_paths=1,
    )
    decoded_dense = tf.sparse_to_dense(
        decoded.indices, decoded.dense_shape, decoded.values, default_value=-1)
    return (decoded_dense,)

def _tf_ctc_dense_to_sparse(y_true, label_lengths):
    import tensorflow as tf
    # y_true (batch_size, max_seq_length)
    # label_lengths (batch_size,)
    dense_shape = tf.shape(y_true)
    dense_mask = tf.sequence_mask(label_lengths, dense_shape[1])
    sparse_values = tf.boolean_mask(tf.to_int32(y_true), dense_mask)
    sparse_indices = tf.where(dense_mask)
    return tf.SparseTensor(
        sparse_indices, sparse_values, tf.to_int64(dense_shape))

def _tf_dft_ctc_loss(y_true, y_pred, input_length, label_length):
    import tensorflow as tf
    # replicate the logic in ctc_batch_cost, sans log and pre softmax
    label_length = tf.to_int32(tf.reshape(label_length, [-1]))
    input_length = tf.to_int32(tf.reshape(input_length, [-1]))
    sparse_labels = _tf_ctc_dense_to_sparse(y_true, label_length)
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    loss = tf.nn.ctc_loss(
        inputs=y_pred, labels=sparse_labels, sequence_length=input_length)
    return tf.expand_dims(loss, 1)

def _tf_warp_ctc_loss(y_true, y_pred, input_length, label_length):
    import tensorflow as tf
    import warpctc_tensorflow
    # replicate the logic in ctc_batch_cost, sans log and pre softmax
    label_length = tf.to_int32(tf.reshape(label_length, [-1]))
    input_length = tf.to_int32(tf.reshape(input_length, [-1]))
    sparse_labels = _tf_ctc_dense_to_sparse(y_true, label_length)
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    with K.get_session().graph._kernel_label_map({"CTCLoss": "WarpCTC"}):
        loss = tf.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length)
    return tf.expand_dims(loss, 1)

if K.backend() == 'tensorflow':
    try:
        import warpctc_tensorflow
        _ctc_loss = _tf_warp_ctc_loss
    except ImportError:
        print('Warp-ctc not installled. Using built-in ctc', file=stderr)
        _ctc_loss = _tf_dft_ctc_loss
    _ctc_decode = _tf_dft_ctc_decode
else:
    _ctc_loss = _dft_ctc_loss
    _ctc_decode = _dft_ctc_decode
