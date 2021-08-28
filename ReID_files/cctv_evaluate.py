#!/usr/bin/env python
# encoding: utf-8

import sys
import os
import glob
from os import path as osp
import cv2
import torch

sys.path.append('.')
print(os.getcwd())

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer

# Path to the data
DATA = '/mnt/sdd2/herta_aimars/datasets-reid/abnormal_identities/'
QUERY_FOLDER = DATA + "query/"
TEST_SET_FOLDER = DATA + "test_set/"

# Get normalized images from query and gallery for each test clip
def get_images_from_test_folder(test_folder, cfg):

    query = []
    test_set_esc = []
    test_set_fac = []

    for test in os.listdir(QUERY_FOLDER):
        if test == test_folder:
            aux = osp.join(osp.join(QUERY_FOLDER, test), "*jpg")
            query_files = glob.glob(aux)
            for img in query_files:
                image = preprocess(cv2.imread(img), cfg)
                query.append(image)

    for test in os.listdir(TEST_SET_FOLDER):
        if test == test_folder:
            for cam in os.listdir(osp.join(TEST_SET_FOLDER, test)):
                aux = osp.join(osp.join(TEST_SET_FOLDER, test, cam), "*jpg")
                if cam == "esc":
                    test_set_esc_files = glob.glob(aux)
                    for img in test_set_esc_files:
                        image = preprocess(cv2.imread(img), cfg)
                        test_set_esc.append(image)
                else:
                    test_set_fac_files = glob.glob(aux)
                    for img in test_set_fac_files:
                        image = preprocess(cv2.imread(img), cfg)
                        test_set_fac.append(image)

    return query, test_set_esc, test_set_fac

# Reshape images
def preprocess(image, cfg):

    # The model expects RGB inputs
    image = image[:, :, ::-1]

    # Apply pre-processing to image
    resized_image = cv2.resize(image, tuple([*cfg.INPUT.SIZE_TEST][::-1]), interpolation = cv2.INTER_CUBIC)

    preproc_image = torch.as_tensor(resized_image.astype("float32").transpose(2, 0, 1))[None]

    #preproc_image = preproc_image - (torch.Tensor(cfg.MODEL.PIXEL_MEAN).reshape((1, -1, 1, 1))) / torch.Tensor(cfg.MODEL.PIXEL_STD).reshape((1, -1, 1, 1))

    return preproc_image

def main():

    # Get configuration from file
    cfg = get_cfg()
    cfg.merge_from_file('/mnt/sdd2/herta_aimars/fastReID/fast-reid-master/configs/CCTV/sbs_MN3.yml')

    # Get best saved model
    model = DefaultTrainer.build_model(cfg)
    Checkpointer(model).load(os.path.join(cfg.OUTPUT_DIR, 'model_best.pth'))

    # Disable all training options
    model.eval()

    # Test
    for test_clip in os.listdir(QUERY_FOLDER):
        query, test_esc, test_fac = get_images_from_test_folder(test_clip, cfg)

        # To tensor
        query_tensor = torch.cat(query)
        test_esc_tensor = torch.cat(test_esc)
        test_fac_tensor = torch.cat(test_fac)

        # Get features
        query_tensor = query_tensor.to('cuda:0')
        test_esc_tensor = test_esc_tensor.to('cuda:0')
        test_fac_tensor = test_fac_tensor.to('cuda:0')
        features_query = model(query_tensor)
        features_test_esc = model(test_esc_tensor)
        features_test_fac = model(test_fac_tensor)

        # Compute cosine similarity
        for query_feat_vector in features_query:
            for esc_feat_vector in features_test_esc:
                cos = torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(query_feat_vector), torch.nn.functional.normalize(esc_feat_vector))

        '''query_esc_cos = torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(features_query, 0), torch.nn.functional.normalize(features_test_esc, 0))
        query_fac_cos = torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(features_query, 0), torch.nn.functional.normalize(features_test_fac, 0))

        # Get max values from cosine similarity between query and esc images
        max_esc = torch.max(query_esc_cos, 1)
        max_esc_indices = max_esc.indices
        max_esc_values = max_exc.values

        # Get max values from cosine similarity between query and fac images
        max_fac = torch.max(query_fac_cos, 1)
        max_fac_indices = max_fac.indices
        max_fac_values = max_fac.values'''

if __name__ == '__main__':
    main()
