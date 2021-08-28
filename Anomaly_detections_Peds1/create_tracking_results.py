# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Copy results to ./tracking_results/
"""

import os
from os import path as osp
import shutil

#Tracking algorithm output folder
source_track = "./FairMOT/demos/peds1/"
#(To work on CCTV dataset, change "./FairMOT/demos/peds1/" to "./FairMOT/demos/cctv_cen/") NOT AVAILABLE

#Folder to store tracking information
target_track = "./tracking_results/"
target_track_train = "./tracking_results/train/"
target_track_test = "./tracking_results/test/"

if not os.path.exists(target_track):
    os.makedirs(target_track)

if not os.path.exists(target_track_train):
    os.makedirs(target_track_train)

    if not os.path.exists(target_track_test):
        os.makedirs(target_track_test)

for file in os.listdir(source_track):
    if os.path.isfile(os.path.join(source_track, file)) and file.endswith(".txt"):

      source = source_track + file

      target = ""

      if (file[1] == "r"):

        target = target_track_train + file

      elif (file[1] == "e"):

        target = target_track_test + file

      shutil.copyfile(source, target)
