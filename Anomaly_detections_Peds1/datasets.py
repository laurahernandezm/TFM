# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Manage datasets
File structure taken from Tracking without bells and whistles (https://github.com/phil-bergmann/tracking_wo_bnw)
"""

from peds1_wrapper import Peds1Wrapper
#from cctv_wrapper import CCTVWrapper NOT AVAILABLE

_sets = {}

#Fill all available datasets, change here to modify / add new datasets.
for split in ['train', 'test']:
    name = f'peds1_{split}'
    _sets[name] = (lambda *args, split=split: Peds1Wrapper(split, *args))

#CCTV dataset NOT AVAILABLE
'''for split in ['train', 'test']:
    name = f'cctv_{split}'
    _sets[name] = (lambda *args, split=split: CCTVWrapper(split, *args))'''


class Datasets(object):
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, dataset, *args):
        """Initialize the corresponding dataloader.

        Keyword arguments:
        dataset --  the name of the dataset
        args -- arguments used to call the dataloader
        """
        assert dataset in _sets, "[!] Dataset not found: {}".format(dataset)

        if len(args) == 0:
            args = [{}]

        self._data = _sets[dataset](*args)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
