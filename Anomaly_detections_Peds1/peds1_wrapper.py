# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Manage datasets
File structure taken from Tracking without bells and whistles (https://github.com/phil-bergmann/tracking_wo_bnw)
"""

import torch
from torch.utils.data import Dataset
from peds1_sequence import Peds1Sequence

class Peds1Wrapper(Dataset):

    def __init__(self, split, dataloader):

        train_sequences = ['Train01', 'Train02', 'Train03', 'Train04', 'Train05',
                            'Train06', 'Train07', 'Train08', 'Train09', 'Train10',
                            'Train11', 'Train12', 'Train13', 'Train14', 'Train15',
                            'Train16', 'Train17', 'Train18', 'Train19', 'Train20',
                            'Train21', 'Train22', 'Train23', 'Train24', 'Train25',
                            'Train26', 'Train27', 'Train28', 'Train29', 'Train30',
                            'Train31', 'Train32', 'Train33', 'Train34', 'Train35',
                            'Train36', 'Train37', 'Train38', 'Train39', 'Train40',
                            'Train41', 'Train42', 'Train43', 'Train44', 'Train45',
                            'Train46', 'Train47', 'Train48', 'Train49', 'Train50']
        test_sequences = ['Test01', 'Test02', 'Test03', 'Test04', 'Test05',
                            'Test06', 'Test07', 'Test08', 'Test09', 'Test10',
                            'Test11', 'Test12', 'Test13', 'Test14', 'Test15',
                            'Test16', 'Test17', 'Test18', 'Test19', 'Test20',
                            'Test21', 'Test22', 'Test23', 'Test24', 'Test25',
                            'Test26', 'Test27', 'Test28', 'Test29', 'Test30',
                            'Test31', 'Test32', 'Test33', 'Test34', 'Test35',
                            'Test36', 'Test37', 'Test38', 'Test39', 'Test40',
                            'Test41', 'Test42', 'Test43', 'Test44', 'Test45',
                            'Test46', 'Test47', 'Test48', 'Test49', 'Test50']

        if "train" == split:
            sequences = train_sequences
        elif "test" == split:
            sequences = test_sequences
        else:
            raise NotImplementedError("Split not available.")

        self._data = []

        for s in sequences:
            self._data.append(Peds1Sequence(seq_name=s, **dataloader))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
