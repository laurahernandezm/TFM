# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Manage datasets
File structure taken from Tracking without bells and whistles (https://github.com/phil-bergmann/tracking_wo_bnw)
"""

import re
import configparser
import csv
import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import cv2

from torchvision.transforms import ToTensor

DATA_DIR = "./data"

class Peds1Sequence(Dataset):
    """Multiple Object Tracking Dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

    def __init__(self, seq_name=None, vis_threshold=0.0, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self.vis_threshold = vis_threshold

        self._mot_dir = osp.join(DATA_DIR, 'Peds1')

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = ToTensor()

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.tensor([det[:4] for det in data['dets']])
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        return sample

    def _sequence(self):
        seq_name = self._seq_name

        if seq_name in self._train_folders:
            seq_path = osp.join(self._mot_dir, 'train', seq_name)
        else:
            seq_path = osp.join(self._mot_dir, 'test', seq_name)

        #im_dir = osp.join(seq_path, 'img1')
        im_dir = seq_path
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')
        det_file = osp.join(seq_path, 'det', 'det.txt')

        total = []

        boxes = {}
        dets = {}
        visibility = {}

        valid_files = [f for f in os.listdir(im_dir) if len(re.findall("[0-9]+.jpg", f)) == 1]
        seq_length = len(valid_files)

        no_gt = False

        for i in range(1, seq_length+1):
            boxes[i] = {}
            dets[i] = []
            visibility[i] = {}


        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    # class person, certainity 1, visibility >= 0.25
                    if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
                        # Make pixel indexes 0-based, should already be 0-based (or not)
                        x1 = int(row[2]) - 1
                        y1 = int(row[3]) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + int(row[4]) - 1
                        y2 = y1 + int(row[5]) - 1
                        bb = np.array([x1,y1,x2,y2], dtype=np.float32)
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    x1 = float(row[2]) - 1
                    y1 = float(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + float(row[4]) - 1
                    y2 = y1 + float(row[5]) - 1
                    score = float(row[6])
                    bb = np.array([x1,y1,x2,y2, score], dtype=np.float32)
                    dets[int(float(row[0]))].append(bb)

        for i in range(1,seq_length+1):
            im_path = osp.join(im_dir,"{:06d}.jpg".format(i))

            sample = {'gt':boxes[i],
                      'im_path':im_path,
                      'vis':visibility[i],
                      'dets':dets[i],}

            total.append(sample)

        return total, no_gt

    def get_det_file(self, label_path, raw_label_path, mot17_label_path):
        if self._dets == "DPM":
            det_file = osp.join(label_path, 'det', 'det.txt')
        elif self._dets == "DPM_RAW16":
            det_file = osp.join(raw_label_path, 'det', 'det-dpm-raw.txt')
        elif "17" in self._seq_name:
            det_file = osp.join(
                mot17_label_path,
                f"{self._seq_name}-{self._dets[:-2]}",
                'det',
                'det.txt')
        else:
            det_file = ""
        return det_file

    def __str__(self):
        return self._seq_name

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = osp.join(output_dir, self._seq_name+'.txt')

        print("[*] Writing to: {}".format(file))

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])
