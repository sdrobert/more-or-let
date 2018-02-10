'''Callbacks and callback-related periphery'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from csv import DictReader
from six.moves.cPickle import dump

import numpy as np

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import History

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

__all__ = [
    'ExtendedHistory',
    'ExtendedEarlyStopping',
    'RandomStateCheckpoint'
]


try:
    FileNotFoundError()
except NameError:
    FileNotFoundError = IOError


class ExtendedHistory(History):
    '''History callback, enhanced

    This callback keeps track of additional metadata and injects some
    information into the log dictionary.

    ``csv_path`` is used to restore the state of the ExtendedHistory
    from a previous run.

    At the beginning of training, ``ExtendedHistory`` adds the following
    info to the log (set to None if not available):

    prev_monitored : str
        The value stored in 'monitored' from previous epochs
    prev_best: float
        The value stored in 'best' from previous epochs
    prev_mode : str
        The value stored in 'mode' from previous epochs
    prev_patience : int
        The value stored in 'patience' (cast to an int) from the last
        epoch
    prev_min_delta : float
        The value stored in 'min_delta' (cast to a float) from the last
        epoch
    prev_wait : int
        The value stored in 'wait' (cast to an int) from the last epoch
    prev_training_stage : str
        The value stored in 'training_stage' from the last epoch
    training_stage : str
        The current setting of 'training_stage'

    At the end of an epoch, `ExtendedHistory` adds the following info to
    the log (set to None if not available):

    training_stage : str
        The current setting of 'training_stage'
    model_path : str
        The model path to save to, whose value is determined by
        formatting ``model_formatter`` with the values in the current
        log

    Additional entries can be added to the log by setting additional
    keyword arguments at intialization. They will be perpetuated by
    ``csv_path``

    .. warning:: Do not add any additional metadata

    Parameters
    ----------
    csv_path : str, optional
        The file to read history from
    strict : bool
        When reading the CSV, whether to sanity check the CSV
    delimiter : str
        The delimiter used to delimit fields in the csv
    model_formatter : str, optional
        The format string used when (if) writing model files
    training_stage : str
        This run's stage of training
    additional_metadata : Keyword arguments, optional
        Additional entries to be added to the log. They are considered
        static over the course of the experiment.
    '''

    def __init__(
            self, csv_path=None, strict=True, delimiter=',',
            model_formatter=None, training_stage='adam',
            **additional_metadata):
        self.strict = strict
        self.delimiter = delimiter
        self.model_formatter = model_formatter
        self.training_stage = training_stage
        self.additional_metadata = additional_metadata
        self.csv_path = csv_path
        self.epoch = None
        self.history = None

    def load_csv_history(self):
        '''Load up the csv history'''
        if self.csv_path is not None:
            self.csv_epoch = -float('inf')
            self.csv_history = dict()
            one_row = False
            try:
                with open(self.csv_path, 'r') as file_obj:
                    reader = DictReader(file_obj, delimiter=self.delimiter)
                    for row in reader:
                        one_row = True
                        epoch = int(row.pop('epoch'))
                        if epoch < self.csv_epoch:
                            continue
                        elif self.strict and epoch == self.csv_epoch:
                            raise ValueError(
                                'Epoch {} occurs twice in csv {}'.format(
                                    epoch, self.csv_path))
                        self.csv_epoch = epoch
                        for key, value in row.items():
                            if value is None and key in self.csv_history:
                                continue  # skip missing entries
                            if key in (
                                    'best', 'min_delta', 'loss',
                                    'val_loss', 'acc', 'val_acc', 'lr'):
                                value = float(value)
                            elif key in ('patience', 'wait'):
                                value = int(value)
                            self.csv_history[key] = value
                if not one_row:
                    raise FileNotFoundError()  # pretend the file doesn't exist
                for key, value in self.additional_metadata.items():
                    if self.strict and key not in self.csv_history:
                        raise ValueError(
                            'The keyword "{}" was present in initialization '
                            'but not in the csv file {}'.format(
                                key, self.csv_path))
                    act_value = self.csv_history[key]
                    if self.strict:
                        try:
                            close = np.isclose(value, act_value)
                        except TypeError:
                            close = (str(value) == str(act_value))
                        if not close:
                            raise ValueError(
                                'Expected "{}" to have the value "{}"; got '
                                '"{}" from csv {}'.format(
                                    key, value, act_value, self.csv_path))
            except FileNotFoundError:
                self.csv_history = self.additional_metadata
        else:
            self.csv_epoch = -float('inf')
            self.csv_history = self.additional_metadata

    def get_last_epoch(self):
        '''Get the last recorded epoch'''
        if self.epoch:
            return self.epoch[-1]
        else:
            self.load_csv_history()
            return self.csv_epoch

    def get_last(self, key):
        '''Get the last recorded value of a specific property'''
        if key == 'epoch':
            return self.get_last_epoch()
        if self.epoch and key in self.history:
            for value in self.history[key][::-1]:
                if value is not None:
                    return value
        self.load_csv_history()
        return self.csv_history.get(key)

    def on_train_begin(self, logs=None):
        logs = logs if logs is not None else dict()
        self.epoch = []
        self.history = dict()
        self.load_csv_history()
        for key in (
                'monitor', 'best', 'mode', 'patience', 'min_delta', 'wait',
                'training_stage'):
            if key in self.csv_history:
                logs['prev_' + key] = self.csv_history[key]
        logs['training_stage'] = self.training_stage

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else dict()
        if epoch in self.epoch:
            raise ValueError('Epoch {} occurred twice!'.format(epoch))
        # inject all logs but model_path
        if 'training_stage' in self.csv_history or self.training_stage:
            logs['training_stage'] = self.training_stage
        for key, value in self.additional_metadata.items():
            logs[key] = value
        # now handle model_path
        if 'model_path' in self.csv_history or self.model_formatter:
            if self.model_formatter:
                logs['model_path'] = self.model_formatter.format(
                    epoch=epoch + 1, **logs)
            else:
                logs['model_path'] = self.model_formatter
        # update history
        self.epoch.append(epoch)
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)


class ExtendedEarlyStopping(EarlyStopping):
    '''Early stopping, enhanced

    This subclass of ``EarlyStopping`` puts its hyperparameters in the
    logs after every epoch. They include

     - the value monitored ("monitor")
     - the "mode", i.e. how to compare monitored values (one of "min",
       "max", or "auto")
     - the best monitored value seen ("best")
     - the minimum-magnitude difference from best that can be considered
       an improvement ("min_delta")
     - "patience", i.e. the number of epochs to wait without improvement
       before early stopping
     - the number of epochs already waited without improvement ("wait")

    When placed after ``ExtendedHistory`` in a callback list, it can
    recover its hyperparameters from previous epochs at the beginning
    of training. These will clobber whatever is specified here on
    initialization.

    Parameters
    ----------
    monitor : str
    min_delta : float
    patience : int
    verbose : int
        Verbosity mode (``verbose > 0`` is verbose)
    mode : {'min', 'max', 'auto'}
        In "min" mode, training will stop when the quantity monitored
        has stopped decreasing; in "max" mode it will stop when the
        quantity monitored has stopped increasing; in "auto" mode,
        the direction is automatically inferred from the name of the
        monitored quantity.
    reset_on_new_training_stage : bool
        If, on_train_begin, the logged "prev_training_stage" has been
        set and does not match the logged "training_stage", the training
        stage will be reset

    Attributes
    ----------
    best : float
    monitor : str
    patience : int
    verbose : int
    min_delta : float
    wait : int
    mode : {'min', 'max'}
    stopped_epoch : int
    reset_on_new_training_stage : bool
    mode : str
    hyperparams : tuple
        A sequence of names of hyperparameters that are stored to logs
        on each epoch end
    '''

    def __init__(
            self, monitor='val_loss', min_delta=0, patience=0, verbose=0,
            mode='auto', reset_on_new_training_stage=True):
        self.reset_on_new_training_stage = reset_on_new_training_stage
        self.hyperparams = (
            'monitor', 'best', 'mode', 'patience', 'min_delta', 'wait')
        super(ExtendedEarlyStopping, self).__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
        )
        self.mode = 'min' if self.monitor_op == np.less else 'max'

    def on_train_begin(self, logs=None):
        logs = logs if logs is not None else dict()
        super(ExtendedEarlyStopping, self).on_train_begin(logs)
        verb_message = ''
        if any(logs.get('prev_' + hp) is None for hp in self.hyperparams):
            verb_message += 'No record of prior early stopping. '
        elif self.reset_on_new_training_stage and (
                logs.get('prev_training_stage') is None or
                logs['prev_training_stage'] != logs.get('training_stage')):
            verb_message += 'New training stage. Resetting early stopping. '
        else:
            verb_message += 'Loading previous early stopping hyperparams. '
            for hyperparam in self.hyperparams:
                setattr(self, hyperparam, logs['prev_' + hyperparam])
            if self.mode == 'min':
                self.monitor_op = np.less
                self.min_delta *= -1  # stored as absolute value
            else:
                self.monitor_op = np.greater
        if self.wait >= self.patience:
            self.stopped_epoch = -1
            self.model.stop_training = True
        if self.verbose > 0:
            for hyperparam in self.hyperparams:
                verb_message += '{}={}, '.format(
                    hyperparam, getattr(self, hyperparam))
            print(verb_message)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else dict()
        super(ExtendedEarlyStopping, self).on_epoch_end(epoch, logs=logs)
        for hyperparam in self.hyperparams:
            logs[hyperparam] = getattr(self, hyperparam)
        # this is made negative when we monitor with 'min', but we want
        # it to look like our setting, so keep it max
        logs['min_delta'] = abs(self.min_delta)


class RandomStateCheckpoint(Callback):
    '''Save the state of numpy's randomizer every epoch

    Parameters
    ----------
    rng_formatter : str
        An expression that can be formatted with the logs that will
        dictate where the state should be saved to
    rng : numpy.random.RandomState, optional
        The ``RandomState`` object to save the state from. If not set,
        numpy's global randomizer will be used (not recommended)
    verbose : {0, 1}, optional
        0: quiet. 1: loud
    log_entry : str, optional
        If set, a log entry will be added with the formatted path of
        this epoch's state with this key

    Attributes
    ----------
    rng_formatter : str
    rng : numpy.random.RandomState or None
    verbose : {0, 1}
    log_entry : str or None
    '''

    def __init__(self, rng_formatter, rng=None, verbose=0, log_entry=None):
        super(RandomStateCheckpoint, self).__init__()
        self.rng_formatter = rng_formatter
        self.rng = rng
        self.verbose = verbose
        self.log_entry = log_entry

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else dict()
        rng_path = self.rng_formatter.format(epoch=epoch + 1, **logs)
        if self.log_entry:
            logs[self.log_entry] = rng_path
        if self.rng is None:
            state = np.random.get_state()
        else:
            state = self.rng.get_state()
        with open(rng_path, 'wb') as rng_file:
            dump(state, rng_file)
        if self.verbose > 0:
            print('\nEpoch {:05d}: saving rng state to {}'.format(
                epoch + 1, rng_path))
