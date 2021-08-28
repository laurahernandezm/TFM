# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Create test folder to compute metrics
"""

import os
from os import path as osp
import shutil

#Tracking algorithm test folder to compute metrics
root_track = "./information/test/"
test_track = "./test_track/"

if not os.path.exists(test_track):
    os.makedirs(test_track)

for file in os.listdir(root_track):
    if os.path.isfile(os.path.join(root_track, file)) and file.endswith("filtered_trajectories_both.txt"):

        source = root_track + file
        target = test_track + file[0:6] + ".txt"
        #(To work on CCTV dataset, change [0:6] to [0:31]) NOT AVAILABLE

        shutil.copyfile(source, target)
