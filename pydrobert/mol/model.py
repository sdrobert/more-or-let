'''Tensor ops and models'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import keras.backend as K
import numpy as np

from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomUniform
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers.merge import Maximum
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from pydrobert.mol.callbacks import ExtendedEarlyStopping
from pydrobert.mol.callbacks import ExtendedHistory
from pydrobert.mol.config import TrainConfig

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"


class ConvCTC(object):
    '''Fully convolutional CTC

    Architecture is based off [1]_ entirely.

    Parameters
    ----------
    config : config.ModelConfig
        A configuration object containing model-building hyperparameters

    Attributes
    ----------
    config : config.ModelConfig
    model : Keras.model
        The underlying keras acoustic model

    .. [1] Zhang, Y et al. "Towards End-to-End Speech Recognition with Deep
       Convolutional Neural Networks" https://arxiv.org/abs/1701.02720
    '''

    def __init__(self, config):
        if K.image_dim_ordering() != 'tf':
            # not sure if I'm right, but I think the TimeDistributed
            # wrapper will always take axis 1, which could be the
            # channel axis in Theano
            raise ValueError('dimensions must be tensorflow-ordered')
        self.config = config
        self.model = None

    def fit_generator(
            self, train_config, train_data, val_data=None):
        '''Fit the acoustic model to data from generators

        Parameters
        ----------
        train_config : pydrobert.mol.config.TrainConfig
        train_data : pydrobert.mol.corpus.TrainData
            Should yield sample tuples of (feats, labels, feat_sizes,
            label_sizes)
        val_data : pydrobert.mol.corpus.EvalData, optional
            Validation data. Should yield sample tuples of (feats,
            labels, feat_sizes, label_sizes)
        '''
        if train_data.batch_size != train_config.batch_size:
            raise ValueError(
                'Expected training data to have batch size {}; got {}'.format(
                    train_config.batch_size, train_data.batch_size))
        elif val_data and val_data.batch_size != train_config.batch_size:
            raise ValueError(
                'Expected val data to have batch size {}; got {}'.format(
                    train_config.batch_size, val_data.batch_size))
        elif (train_data.num_sub != 4) or (
                train_data.axis_lengths != ((0, 0), (1, 0))):
            raise ValueError(
                'Expecting training data to yield sub-samples of '
                '(feats, label, feat_len, label_len)')
        elif val_data and ((val_data.num_sub != 4) or (
                val_data.axis_lengths != ((0, 0), (1, 0)))):
            raise ValueError(
                'Expecting val data to yield sub-samples of '
                '(feats, label, feat_len, label_len)')
        elif any(axis != 0 for axis in train_data.batch_axis):
            raise ValueError('All batch axes in training data must be 0!')
        elif val_data and any(axis != 0 for axis in val_data.batch_axis):
            raise ValueError('All batch axes in validation data must be 0!')
        # create history callback
        additional_metadata = vars(self.config)
        additional_metadata.update(vars(train_config))
        # training_stage and patience get added by callbacks
        # the rest don't affect the model directly
        del additional_metadata['training_stage']
        del additional_metadata['csv_path']
        del additional_metadata['csv_delimiter']
        del additional_metadata['patience']
        del additional_metadata['model_formatter']
        history = ExtendedHistory(
            csv_path=train_config.csv_path,
            strict=True,
            delimiter=train_config.csv_delimiter,
            model_formatter=train_config.model_formatter,
            training_stage=train_config.training_stage,
            **additional_metadata
        )
        initial_epoch = max(-1, history.get_last_epoch()) + 1
        # get acoustic model ready
        model_path = history.get_last_model_path()
        prev_training_stage = history.get_last_training_stage()
        self._ready_acoustic_model(
            model_path=model_path,
            train_config=train_config,
            prev_training_stage=prev_training_stage,
        )
        if val_data:
            val_monitored = 'val_loss'
        else:
            val_monitored = 'loss'
        callbacks = [history]
        if train_config.patience is not None:
            callbacks.append(ExtendedEarlyStopping(
                monitor=val_monitored,
                min_delta=train_config.min_delta,
                patience=train_config.patience,
                mode='min',
            ))
        if train_config.model_formatter:
            callbacks.append(ModelCheckpoint(
                monitor=val_monitored,
                filepath=train_config.model_formatter,
                save_best_only=False,
                period=1,
            ))
        if train_config.csv_path:
            callbacks.append(CSVLogger(
                train_config.csv_path,
                separator=train_config.csv_delimiter,
                append=initial_epoch,
            ))
        self.model.fit_generator(
            train_data.batch_generator(repeat=True),
            steps_per_epoch=len(train_data),
            epochs=train_config.max_epochs,
            callbacks=callbacks,
            validation_data=(
                val_data.batch_generator(repeat=True) if val_data else None),
            validation_steps=len(val_data) if val_data else 0,
            initial_epoch=initial_epoch,
            shuffle=False,
        )

    def decode(self, decode_config, feats):
        '''Decode features

        Parameters
        ----------
        decode_config : pydrobert.mol.config.DecodeConfig
        feats : 2D float32
            Of shape `(frame, feat)` representing a single recording

        Returns
        -------
        tuple
            Integer id sequence representing the labels assigned to the
            sequence
        '''
        self._ready_acoustic_model(model_path=decode_config.model_path)
        decoder = self._construct_decoder(decode_config.beam_width)
        length_batch = np.asarray([[feats.shape[0]]], dtype=np.int32)
        ret_labels = decoder(
            [feats[np.newaxis, :, :, np.newaxis], length_batch, 0]
        )[0][0]
        return tuple(int(idee) for idee in ret_labels if idee != -1)

    def decode_generator(self, decode_config, eval_data):
        '''Decode features from generator

        Parameters
        ----------
        decode_config : pydrobert.mol.config.DecodeConfig
        eval_data : pydrobert.mol.corpus.EvalData
            Evaluation/test data. Should yield samples which are one of:
                1. only features
                2. (key, feats)
                3. (feats, feat_len)
                4. (key, feats, feat_len)
            3. and 4. are preferred as they allow batches of greater
            than size 1 to be processed.

        Yields
        ------
        tuple
            batches which are one of:
                1. [seq]
                2. [(key, seq)]
                3. [seq] * eval_data.batch_size
                4. [(key, seq)] * eval_data.batch_size
            according to what eval_data yields. The outermost length is
            the length of the batch (1. and 2. correspond to batch
            sizes of 1, 3. and 4. of eval_data.batch_size). seq is an
            integer id tuple representing the labels assigned to the
            associated feature.
        '''
        # determine the eval_data setup
        eval_style = None
        if eval_data.add_key:
            if eval_data.num_sub == 3 and eval_data.axis_lengths == ((0, 0),):
                eval_style = 4
            elif eval_data.num_sub == 2:
                eval_style = 2
        elif eval_data.num_sub == 2 and eval_data.axis_lengths == ((0, 0),):
            eval_style = 3
        elif eval_data.num_sub == 1:
            eval_style = 1
        if eval_style is None:
            raise ValueError(
                'Expected evaluation data to yield samples of one of: '
                'feats, (key, feats), (feats, feat_len), or (key, feats, '
                'feat_len)')
        self._ready_acoustic_model(model_path=decode_config.model_path)
        decoder = self._construct_decoder(decode_config.beam_width)
        if eval_style <= 2:
            for sample in eval_data.sample_generator():
                if eval_style == 1:
                    feat_len = sample.shape[0]
                    ret_labels = decoder([
                        sample[np.newaxis, :, :, :],  # add batch_axis
                        np.array([feat_len], dtype=np.int32),
                        0,
                    ])[0][0]
                    yield [tuple(
                        int(idee) for idee in ret_labels if idee != -1)]
                else:
                    key, feats = sample
                    feat_len = feats.shape[0]
                    ret_labels = decoder([
                        feats[np.newaxis, :, :, :],
                        np.array([feat_len], dtype=np.int32),
                        0,
                    ])[0][0]
                    yield [(key, tuple(
                        int(idee) for idee in ret_labels if idee != -1))]
        else:
            for batch in eval_data.batch_generator():
                if eval_style == 3:
                    ret_labels = decoder(batch + (0,))[0]
                    yield [
                        tuple(int(idee) for idee in sample if idee != -1)
                        for sample in ret_labels
                    ]
                else:
                    ret_labels = decoder(batch[1:] + (0,))[0]
                    yield [
                        (
                            key,
                            tuple(int(idee) for idee in sample if idee != -1),
                        )
                        for key, sample in zip(batch[0], ret_labels)
                    ]

    def _ready_acoustic_model(
            self, model_path=None, train_config=None,
            prev_training_stage=None):
        # 3 options exist to get an acoustic model
        # 1. load a model verbatim from file
        # 2. construct a model from scratch and initialize
        # 3. load a model's weights from file and use them in a new
        #    model
        # option 1. is for resuming training in the same stage or we're
        # decoding. Option 2 is for when we are just starting training
        # or there's no model to load. Option 3 is when we're switching
        # training stages.
        if model_path and (
                train_config is None or
                train_config.training_stage == prev_training_stage):
            self.model = load_model(
                model_path, custom_objects={
                    '_ctc_loss': _ctc_loss,
                    '_y_pred_loss': _y_pred_loss,
                },
            )
            return  # assume already compiled
        if train_config is None:
            train_config = TrainConfig()
        self._construct_acoustic_model(train_config)
        if model_path:
            self.model.load_weights(model_path, by_name=True)
        if train_config.training_stage == 'adam':
            optimizer = Adam(lr=train_config.adam_lr, clipvalue=1.0)
        elif train_config.training_stage == 'sgd':
            optimizer = SGD(lr=train_config.sgd_lr, clipvalue=1.0)
        self.model.compile(
            loss={'ctc_loss': _y_pred_loss},
            optimizer=optimizer,
        )

    def _construct_acoustic_model(self, train_config=TrainConfig()):
        # construct an acoustic model from scratch
        layer_kwargs = {
            'activation': 'linear',
            'kernel_initializer': RandomUniform(
                minval=-self.config.weight_init_mag,
                maxval=self.config.weight_init_mag,
                seed=self.config.weight_seed),
        }
        if train_config.training_stage == 'sgd':
            layer_kwargs['kernel_regularizer'] = l2(train_config.sgd_reg)
        # convolutional layer pattern

        def _conv_maxout_layer(last_layer, n_filts, name_prefix, dropout=True):
            conv_a = Conv2D(
                n_filts,
                (self.config.filt_time_width, self.config.filt_freq_width),
                strides=(
                    self.config.filt_time_stride,
                    self.config.filt_freq_stride,
                ),
                padding='same',
                name=name_prefix + '_a',
                **layer_kwargs
            )(last_layer)
            conv_b = Conv2D(
                n_filts,
                (self.config.filt_time_width, self.config.filt_freq_width),
                strides=(
                    self.config.filt_time_stride,
                    self.config.filt_freq_stride,
                ),
                padding='same',
                name=name_prefix + '_b',
                **layer_kwargs
            )(last_layer)
            last = Maximum(name=name_prefix + '_m')([conv_a, conv_b])
            # pre-weights (i.e. post max), as per
            # http://jmlr.org/proceedings/papers/v28/goodfellow13.pdf
            if dropout:
                last = Dropout(
                    train_config.dropout, name=name_prefix + '_d')(last)
            return last
        # inputs
        feat_input = Input(
            shape=(None, self.config.num_feats, 1),
            name='feat_in',
        )
        feat_size_input = Input(
            shape=(1,), dtype='int32', name='feat_size_in')
        label_input = Input(
            shape=(None,), dtype='int32', name='label_in')
        label_size_input = Input(
            shape=(1,), dtype='int32', name='label_size_in')
        last_layer = feat_input
        # convolutional layers
        n_filts = self.config.init_num_filt_channels
        last_layer = _conv_maxout_layer(
            last_layer, n_filts, 'conv_1', dropout=False)
        last_layer = MaxPooling2D(
            pool_size=(
                self.config.pool_time_width,
                self.config.pool_freq_width),
            name='conv_1_p')(last_layer)
        last_layer = Dropout(train_config.dropout, name='conv_1_d')(last_layer)
        for layer_no in range(2, 11):
            if layer_no == 5:
                n_filts *= 2
            last_layer = _conv_maxout_layer(
                last_layer, n_filts, 'conv_{}'.format(layer_no))
        last_layer = Lambda(
            lambda layer: K.max(layer, axis=2),
            output_shape=(None, n_filts),
            name='max_freq_into_channel',
        )(last_layer)
        # dense layers
        for layer_no in range(1, 4):
            name_prefix = 'dense_{}'.format(layer_no)
            dense_a = Dense(
                self.config.num_dense_hidden, name=name_prefix + '_a',
                **layer_kwargs
            )
            dense_b = Dense(
                self.config.num_dense_hidden, name=name_prefix + '_b',
                **layer_kwargs
            )
            td_a = TimeDistributed(
                dense_a, name=name_prefix + '_td_a')(last_layer)
            td_b = TimeDistributed(
                dense_b, name=name_prefix + '_td_b')(last_layer)
            last_layer = Maximum(name=name_prefix + '_m')([td_a, td_b])
            last_layer = Dropout(
                train_config.dropout, name=name_prefix + '_d'
            )(last_layer)
        activation_dense = Dense(
            self.config.num_labels, name='dense_activation',
            **layer_kwargs
        )
        activation_layer = TimeDistributed(
            activation_dense, name='dense_activation_td')(last_layer)
        # we take a page from the image_ocr example and treat the ctc as a
        # lambda layer.
        loss_layer = Lambda(
            lambda args: _ctc_loss(*args),
            output_shape=(1,), name='ctc_loss'
        )([
            label_input,
            activation_layer,
            feat_size_input,
            label_size_input
        ])
        self.model = Model(
            inputs=[
                feat_input,
                label_input,
                feat_size_input,
                label_size_input,
            ],
            outputs=[loss_layer],
        )
        if self.config.num_gpus > 1:
            self.model = multi_gpu_model(self.model, self.config.num_gpus)

    def _construct_decoder(self, beam_width):
        label_out = _ctc_decode(
            self.model.get_layer(name='dense_activation_td').output,
            self.model.get_layer(name='feat_size_in').output,
            beam_width=beam_width
        )
        decoder = K.function(
            [
                self.model.get_layer(name='feat_in').output,
                self.model.get_layer(name='feat_size_in').output,
                K.learning_phase(),
            ],
            label_out,
        )
        return decoder


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
    input_length = tf.reshape(input_length, [-1])
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
    sparse_values = tf.boolean_mask(y_true, dense_mask)
    sparse_indices = tf.where(dense_mask)
    return tf.SparseTensor(
        sparse_indices, sparse_values, tf.to_int64(dense_shape))


def _tf_dft_ctc_loss(y_true, y_pred, input_length, label_length):
    import tensorflow as tf
    # replicate the logic in ctc_batch_cost, sans log and pre softmax
    label_length = tf.reshape(label_length, [-1])
    input_length = tf.reshape(input_length, [-1])
    sparse_labels = _tf_ctc_dense_to_sparse(y_true, label_length)
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    loss = tf.nn.ctc_loss(
        inputs=y_pred, labels=sparse_labels, sequence_length=input_length)
    return tf.expand_dims(loss, 1)


def _tf_warp_ctc_loss(y_true, y_pred, input_length, label_length):
    import tensorflow as tf
    import warpctc_tensorflow
    # replicate the logic in ctc_batch_cost, sans log and pre softmax
    label_length = tf.reshape(label_length, [-1])
    input_length = tf.reshape(input_length, [-1])
    sparse_labels = _tf_ctc_dense_to_sparse(y_true, label_length)
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    with K.get_session().graph._kernel_label_map({"CTCLoss": "WarpCTC"}):
        loss = tf.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length)
    return tf.expand_dims(loss, 1)


def _y_pred_loss(y_true, y_pred):
    '''Simply return y_pred as loss'''
    return y_pred


if K.backend() == 'tensorflow':
    try:
        import warpctc_tensorflow
        _ctc_loss = _tf_warp_ctc_loss
    except ImportError:
        print('Warp-ctc not installled. Using built-in ctc', file=sys.stderr)
        _ctc_loss = _tf_dft_ctc_loss
    _ctc_decode = _tf_dft_ctc_decode
else:
    _ctc_loss = _dft_ctc_loss
    _ctc_decode = _dft_ctc_decode
