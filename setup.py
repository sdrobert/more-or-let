# pylint: skip-file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

setup(
    name='pydrobert-mol',
    description='more-or-let commands and models',
    author=__author__,
    author_email=__email__,
    license=__license__,
    namespace_packages=['pydrobert'],
    packages=['pydrobert', 'pydrobert.mol'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    entry_points={
        'console_scripts': [
            'compute-per = pydrobert.mol.command_line:compute_per',
            'decode-cnn-ctc = pydrobert.mol.command_line:decode_cnn_ctc',
            'train-cnn-ctc = pydrobert.mol.command_line:train_cnn_ctc',
            'normalize-feat-lens = pydrobert.mol.command_line:'
            'normalize_feat_lens',
            'find-best-model-from-log = pydrobert.mol.command_line:'
            'find_best_model_from_log',
        ]
    }
)
