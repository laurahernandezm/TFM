# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Track all train or test videos
"""

import os

#Root data folder
dir = "../../data/Peds1/"

#Choose train or test videos to track
mod = dir + "train/"
#mod = dir + "test/"

for folder in os.listdir(mod):
    f = mod + folder + "/"
    for file in os.listdir(f):
        if (file.endswith(".avi")):
            input_video = f + file

            #Call demo.py with model crowdhuman_dla34.pth and confidence threshold = 0.3
            shell = "python demo.py mot --load_model ../models/crowdhuman_dla34.pth --conf_thres 0.3 --input-video " + input_video
            os.system(shell)
