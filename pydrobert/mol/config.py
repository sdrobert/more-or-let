'''Configuration-related classes and utilities'''

from argparse import ArgumentError

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"


class Config(object):
    '''A generic configuration container

    This parent class uses inflection to determine its public
    attributes. Then it clobbers those attributes with any keyword
    arguments passed to it that share those names.
    '''

    def __init__(self, **kwargs):
        for attr_name in vars(self):
            if attr_name in kwargs:
                setattr(self, attr_name, kwargs[attr_name])

    def add_arguments_to_parser(self, parser, conflict='ignore'):
        '''Add attributes as arguments to this parser

        Current attribute values will be used as defaults and their
        types will be passed as argument types (None is treated as
        a string)

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The parser
        conflict : {'ignore', 'raise'}
            When a name already exists, either leave it be ('ignore')
            or raise an ArgumentError
        '''
        prefix_char = parser.prefix_chars[0]
        for dest in vars(self):
            name = prefix_char * 2 + dest.replace('_', prefix_char)
            default = getattr(self, dest)
            try:
                if default is None:
                    parser.add_argument(name, dest=dest, default=default)
                else:
                    parser.add_argument(
                        name, type=type(default), dest=dest, default=default)
            except ArgumentError:
                if conflict == 'raise':
                    raise


class ModelConfig(Config):
    '''A configuration container for ConvCTC models

    Only the parameters necessary for defining/initializing the
    model.

    Parameters/Attributes
    ---------------------
    num_labels : int
        Number of unique labels, including blank. Defaults to 61
    num_feats : int
        Number of features per recording. Defaults to 123
    delta_order : int
        Number of deltas to concatenate along the feature axis. Defaults
        to 0
    cmvn_rxfilename : str
        If specified, CMVN will be applied to each utterance using the
        stats accumulated to `cmvn_rxfilename`
    init_num_filt_channels : int
        The initial number of filter channels per convolutional
        layer. Defaults to 64
    num_dense_hidden : int
        The number of hidden units in a dense layer. Defaults to
        612
    filt_time_width : int
        The width of convolutional filters along the time axis.
        Defaults to 5
    filt_freq_width : int
        The width of convolutional filters along the frequency
        axis. Defaults to 3
    filt_time_stride : int
        The stride of convolutional filters along the time axis.
        Equivalent to downsampling by this factor. Defaults to 1
    filt_freq_stride : int
        The stride of convolutional filters along the frequency
        axis. Equivalent to downsampling by this factor. Defaults
        to 1
    pool_time_width : int
        The width of Maxout pooling in time. Defaults to 1
    pool_freq_width : int
        The width of Maxout pooling in frequency. Defaults to 3
    weight_init_mag : float
        Magnitude of the range of uniform weight initialization
        (i.e. between [-weight_init_mag, weight_init_mag]). Defaults
        to .05
    weight_seed : int
        Seed to used when initializing weights. Defaults to 1234
    '''

    def __init__(self, **kwargs):
        self.num_labels = 61
        self.num_feats = 123
        self.delta_order = 0
        self.cmvn_rxfilename = ""
        self.init_num_filt_channels = 64
        self.num_dense_hidden = 512
        self.filt_time_width = 5
        self.filt_freq_width = 3
        self.filt_time_stride = 1
        self.filt_freq_stride = 1
        self.pool_time_width = 1
        self.pool_freq_width = 3
        self.weight_init_mag = .05
        self.weight_seed = 1234
        super(ModelConfig, self).__init__(**kwargs)


class TrainConfig(Config):
    '''A configuration container for training ConvCTC models

    Parameters/Attributes
    ---------------------
    training_stage : {'adam', 'sgd'}
        The training stage. 'adam' uses Adam optimization and no
        regularizers. SGD uses stochastic gradient descent with
        an l2 regularizer. Defaults to 'adam'
    csv_path : str
        Path to CSV log file. Defaults to None
    csv_delimiter : str
        Delimiter used in CSV log file. Defaults to ','
    adam_lr : float
        Learning rate used during 'adam' stage. Defaults to 1e-4
    sgd_lr : float
        Learning rate used during 'sgd' stage. Defaults to 1e-5
    sgd_reg : float
        l2 regularization during 'sgd' stage. Defaults to 1e-5
    model_formatter : str
        String used to format model save files. Defaults to
        'model.{epoch:03d}.h5'
    dropout : float
        The probability that a hidden unit is dropped out during
        training. Defaults to .3
    batch_size : int
        The size of batches to generate. Defaults to 20
    patience : int
        The patience of early stopping (number of epochs without
        improvement before quitting). Defaults to 50
    min_delta : float
        The minimum magnitude difference considered to be an
        improvement for early stopping. Defaults to .1
    max_epochs : int
        Total number of epochs to run for, assuming early stopping
        doesn't kick in. Defaults to 1000
    cache : bool
        Whether to cache all training data in memory. Defaults to False
    train_seed : int
        Seed used to initialize training data shuffler at first epoch.
        Defaults to 5678
    train_formatter : str
        String used to format (numpy) rng save files for training.
        Defaults to 'train.{epoch:03d}.pkl'
    '''

    def __init__(self, **kwargs):
        self.csv_path = ""
        self.csv_delimiter = ','
        self.training_stage = 'adam'
        self.adam_lr = 1e-4
        self.sgd_lr = 1e-5
        self.sgd_reg = 1e-5
        self.model_formatter = 'model.{epoch:03d}.h5'
        self.dropout = .3
        self.batch_size = 20
        self.patience = 50
        self.min_delta = .1
        self.max_epochs = 1000
        self.cache = False
        self.train_seed = 5678
        self.train_formatter = 'train.{epoch:03d}.pkl'
        super(TrainConfig, self).__init__(**kwargs)


class DecodeConfig(Config):
    '''A configuration container for decoding with ConvCTC models

    Parameters/Attributes
    ---------------------
    model_path : str
        Path to saved model to load. Defaults to None (but should
        really be set!)
    beam_width : int
        The beam width to apply during decoding. Defaults to 1
    batch_size : int
        The size of batches to generate. Defaults to 20
    '''

    def __init__(self, **kwargs):
        self.model_path = ""
        self.beam_width = 1
        self.batch_size = 20
        super(DecodeConfig, self).__init__(**kwargs)
