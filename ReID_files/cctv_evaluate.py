#!/usr/bin/env python
# encoding: utf-8
"""
Laura Hernández Muñoz

ReID on abnormal test videos
"""

import sys
import os
import glob
from os import path as osp
import cv2
import torch
import torch.nn.functional as F

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer

# Path to the data
ROOT = '/mnt/sdd2/herta_aimars/'
DATA = ROOT + 'datasets-reid/abnormal_identities/'
QUERY_FOLDER = DATA + "query/"
TEST_SET_FOLDER = DATA + "test_set/"

# Threshold to consider a matching
REID_THRES = 0.4

#------------------------------------------------------------------------------#
# Get normalized images from query and gallery for each test clip
#------------------------------------------------------------------------------#
def get_images_from_test_folder(test_folder, cfg):

    query = []
    test_set_esc = []
    test_set_fac = []

    for test in sorted(os.listdir(QUERY_FOLDER)):
        if test == test_folder:
            aux = osp.join(osp.join(QUERY_FOLDER, test), "*jpg")
            query_files = sorted(glob.glob(aux))
            for img in query_files:
                image = preprocess(cv2.imread(img), cfg)
                query.append(image)

    for test in sorted(os.listdir(TEST_SET_FOLDER)):
        if test == test_folder:
            for cam in os.listdir(osp.join(TEST_SET_FOLDER, test)):
                aux = osp.join(osp.join(TEST_SET_FOLDER, test, cam), "*jpg")
                if cam == "esc":
                    test_set_esc_files = sorted(glob.glob(aux))
                    for img in test_set_esc_files:
                        image = preprocess(cv2.imread(img), cfg)
                        test_set_esc.append(image)
                else:
                    test_set_fac_files = sorted(glob.glob(aux))
                    for img in test_set_fac_files:
                        image = preprocess(cv2.imread(img), cfg)
                        test_set_fac.append(image)

    return query, test_set_esc, test_set_fac

#------------------------------------------------------------------------------#
# Reshape images
#------------------------------------------------------------------------------#
def preprocess(image, cfg):

    # The model expects RGB inputs, but we have BGR
    image = image[:, :, ::-1]

    # Apply pre-processing to image
    resized_image = cv2.resize(image, tuple([*cfg.INPUT.SIZE_TEST][::-1]),
                                interpolation = cv2.INTER_CUBIC)

    preproc_image = torch.as_tensor(resized_image.astype("float32").
                                                    transpose(2, 0, 1))[None]

    return preproc_image

#------------------------------------------------------------------------------#
# Count how many correct matches we have in each camera
#------------------------------------------------------------------------------#
def check_matches(test_folder, sims_esc, sims_fac):

    query_ids = []
    correct_matches_esc = []
    correct_matches_fac = []

    # Save query images ID
    for test in sorted(os.listdir(QUERY_FOLDER)):
        if test == test_folder:
            for img in sorted(os.listdir(osp.join(QUERY_FOLDER, test))):
                id = img.split("-")[0][4:]
                query_ids.append(id)

    for test in sorted(os.listdir(TEST_SET_FOLDER)):
        if test == test_folder:
            for cam in os.listdir(osp.join(TEST_SET_FOLDER, test)):
                imgs = sorted(os.listdir(osp.join(TEST_SET_FOLDER, test, cam)))
                for index in range(len(query_ids)):
                    if cam == "esc":
                        # If cosine similarity is higher than REID_THRES and the
                        # IDs match, we have a correct match
                        if sims_esc[index] != -1 and query_ids[index] == \
                        imgs[sims_esc[index]].split("-")[0][4:]:
                            correct_matches_esc.append(True)
                        else:
                            correct_matches_esc.append(False)
                    else:
                        if sims_fac[index] != -1 and query_ids[index] == \
                        imgs[sims_fac[index]].split("-")[0][4:]:
                            correct_matches_fac.append(True)
                        else:
                            correct_matches_fac.append(False)

    return correct_matches_esc, correct_matches_fac

#------------------------------------------------------------------------------#
# Compute TP, TN, FN and FP (with available and not available matches) for each
# camera
# TP == cosine similarity >= REID_THRES and IDs from query and the match are the
#       same (tp_x)
# TN == cosine similarity < REID_THRES and there is no possible match between
#       query and the current camera (tn_x)
# FN == cosine similarity < REID_THRES and there is possible match between query
#       and the current camera (fn_x)
# FP == cosine similarity >= REID_THRES and IDs from query and the match are not
#       the same. We distinguish two possibilities:
#           - The real match is available (fp_av_x)
#           - There is no real match available (fp_not_av_x)
#------------------------------------------------------------------------------#
def confusion_matrix(test_folder, sims_esc, sims_fac):

    query_ids = []
    esc_ids = []
    fac_ids = []

    tp_esc = 0
    tn_esc = 0
    fp_not_av_esc = 0
    fp_av_esc = 0
    fn_esc = 0

    tp_fac = 0
    tn_fac = 0
    fp_not_av_fac = 0
    fp_av_fac = 0
    fn_fac = 0

    # Save query images ID
    for test in sorted(os.listdir(QUERY_FOLDER)):
        if test == test_folder:
            for img in sorted(os.listdir(osp.join(QUERY_FOLDER, test))):
                id = img.split("-")[0][4:]
                query_ids.append(id)

    # Save gallery images ID
    for test in sorted(os.listdir(TEST_SET_FOLDER)):
        if test == test_folder:
            for cam in os.listdir(osp.join(TEST_SET_FOLDER, test)):
                for img in sorted(os.listdir(osp.join(TEST_SET_FOLDER, test, cam))):
                    id = img.split("-")[0][4:]
                    if cam == "esc":
                        esc_ids.append(id)
                    else:
                        fac_ids.append(id)

    # Lists to store the path of the correct matches for facial recognition
    esc_matches_paths = []
    fac_matches_paths = []

    # Lists to store the path of false positive matches for facial recognition
    esc_FP_matches_paths = []
    fac_FP_matches_paths = []

    # Compute TP, TN, FN and FP for esc and fac cameras
    for test in sorted(os.listdir(TEST_SET_FOLDER)):
        if test == test_folder:
            for cam in os.listdir(osp.join(TEST_SET_FOLDER, test)):
                imgs = sorted(os.listdir(osp.join(TEST_SET_FOLDER, test, cam)))
                for index in range(len(query_ids)):
                    if cam == "esc":
                        if sims_esc[index] == -1:
                            if query_ids[index] in esc_ids:
                                fn_esc += 1
                            else:
                                tn_esc += 1
                        else:
                            if query_ids[index] == imgs[sims_esc[index]].split("-")[0][4:]:
                                tp_esc += 1
                                esc_matches_paths.append(osp.join(TEST_SET_FOLDER, test, cam, imgs[sims_esc[index]]))
                            else:
                                if query_ids[index] in esc_ids:
                                    fp_av_esc += 1
                                else:
                                    fp_not_av_esc += 1

                                esc_FP_matches_paths.append(osp.join(TEST_SET_FOLDER, test, cam, imgs[sims_esc[index]]))
                    else:
                        if sims_fac[index] == -1:
                            if query_ids[index] in fac_ids:
                                fn_fac += 1
                            else:
                                tn_fac += 1
                        else:
                            if query_ids[index] == imgs[sims_fac[index]].split("-")[0][4:]:
                                tp_fac += 1
                                fac_matches_paths.append(osp.join(TEST_SET_FOLDER, test, cam, imgs[sims_fac[index]]))
                            else:
                                if query_ids[index] in fac_ids:
                                    fp_av_fac += 1
                                else:
                                    fp_not_av_fac += 1

                                fac_FP_matches_paths.append(osp.join(TEST_SET_FOLDER, test, cam, imgs[sims_fac[index]]))

    print("\n\tTP in esc camera: " + str(tp_esc))
    print("\tTN in esc camera: " + str(tn_esc))
    print("\tFP with available match in esc camera: " + str(fp_av_esc))
    print("\tFP with not available match in esc camera: " + str(fp_not_av_esc))
    print("\tFN in esc camera: " + str(fn_esc))

    print("\n\tTP in fac camera: " + str(tp_fac))
    print("\tTN in fac camera: " + str(tn_fac))
    print("\tFP with available match in fac camera: " + str(fp_av_fac))
    print("\tFP with not available match in fac camera: " + str(fp_not_av_fac))
    print("\tFN in fac camera: " + str(fn_fac))

    return tp_esc, tn_esc, fp_av_esc, fp_not_av_esc, fn_esc, tp_fac, tn_fac, \
    fp_av_fac, fp_not_av_fac, fn_fac, esc_matches_paths, fac_matches_paths, \
    esc_FP_matches_paths, fac_FP_matches_paths

#------------------------------------------------------------------------------#
# Get best saved model, extract features from query and gallery, compute cosine
# similarity and check if matches are correct
#------------------------------------------------------------------------------#
def main():

    # Get configuration from file
    cfg = get_cfg()
    cfg.merge_from_file('/mnt/sdd2/herta_aimars/fastReID/fast-reid-master/configs/CCTV/sbs_MN3.yml')

    # Get best saved model
    model = DefaultTrainer.build_model(cfg)
    Checkpointer(model).load(os.path.join(cfg.OUTPUT_DIR, 'model_best.pth'))

    # Disable all training options
    model.eval()

    total_tp_esc = 0
    total_tn_esc = 0
    total_fp_not_av_esc = 0
    total_fp_av_esc = 0
    total_fn_esc = 0

    total_tp_fac = 0
    total_tn_fac = 0
    total_fp_not_av_fac = 0
    total_fp_av_fac = 0
    total_fn_fac = 0

    esc_facial_paths = []
    fac_facial_paths = []

    esc_FP_facial_paths = []
    fac_FP_facial_paths = []

    # For each test video
    for test_clip in sorted(os.listdir(QUERY_FOLDER)):

        # Get images
        query, test_esc, test_fac = get_images_from_test_folder(test_clip, cfg)

        print("\nCurrent clip: " + str(test_clip))

        # To tensor
        query_tensor = torch.cat(query)
        test_esc_tensor = torch.cat(test_esc)
        test_fac_tensor = torch.cat(test_fac)

        # Send images to GPU and get their features
        query_tensor = query_tensor.to('cuda:0')
        test_esc_tensor = test_esc_tensor.to('cuda:0')
        test_fac_tensor = test_fac_tensor.to('cuda:0')
        features_query = model(query_tensor)
        features_test_esc = model(test_esc_tensor)
        features_test_fac = model(test_fac_tensor)

        max_cos_sims_esc = []
        # Compute cosine similarity between query and test_set_esc
        for query_feat_vector in features_query:
            cos_sims = []
            for esc_feat_vector in features_test_esc:
                cos = F.cosine_similarity(F.normalize(query_feat_vector, dim = -1),
                                        F.normalize(esc_feat_vector, dim = -1),
                                        dim = 0)
                cos_sims.append(cos.item())

            # Get more similar item and save it if its cosine similarity is
            # above the threshold
            max_sim = max(cos_sims)
            if max_sim >= REID_THRES:
                max_cos_sims_esc.append(cos_sims.index(max_sim))
            else:
                max_cos_sims_esc.append(-1)

        max_cos_sims_fac = []
        # Compute cosine similarity between query and test_set_fac
        for query_feat_vector in features_query:
            cos_sims = []
            for fac_feat_vector in features_test_fac:
                cos = F.cosine_similarity(F.normalize(query_feat_vector, dim = -1),
                                        F.normalize(fac_feat_vector, dim = -1),
                                        dim = 0)
                cos_sims.append(cos.item())

            # Get more similar item and save it if its cosine similarity is
            # above the threshold
            max_sim = max(cos_sims)
            if max_sim >= REID_THRES:
                max_cos_sims_fac.append(cos_sims.index(max_sim))
            else:
                max_cos_sims_fac.append(-1)

        # Count how many correct matches there is between query and each camera
        correct_esc, correct_fac = check_matches(test_clip, max_cos_sims_esc, max_cos_sims_fac)

        print("\n\tCorrect matches against esc camera: " +
                str(sum(correct_esc)) + "/" + str(len(correct_esc)))
        print("\tCorrect matches against fac camera: " +
                str(sum(correct_fac)) + "/" + str(len(correct_fac)))

        # Get TP, TN, FN and FP from each camera
        tp_esc, tn_esc, fp_av_esc, fp_not_av_esc, fn_esc, tp_fac, tn_fac, \
        fp_av_fac, fp_not_av_fac, fn_fac, esc_matches_paths, \
        fac_matches_paths, esc_FP_matches_paths, fac_FP_matches_paths = \
        confusion_matrix(test_clip, max_cos_sims_esc, max_cos_sims_fac)

        # Compute total number of TP, TN, FN and FP
        total_tp_esc += tp_esc
        total_tn_esc += tn_esc
        total_fp_not_av_esc += fp_not_av_esc
        total_fp_av_esc += fp_av_esc
        total_fn_esc += fn_esc

        total_tp_fac += tp_fac
        total_tn_fac += tn_fac
        total_fp_not_av_fac += fp_not_av_fac
        total_fp_av_fac += fp_av_fac
        total_fn_fac += fn_fac

        # Save paths of the correct matches to send them to facial recognition module
        esc_facial_paths.extend(esc_matches_paths)
        fac_facial_paths.extend(fac_matches_paths)

        # Save paths of the false positive matches to send them to facial recognition module
        esc_FP_facial_paths.extend(esc_FP_matches_paths)
        fac_FP_facial_paths.extend(fac_FP_matches_paths)

    print("\n\nTotal TP in esc camera: " + str(total_tp_esc))
    print("Total TN in esc camera: " + str(total_tn_esc))
    print("Total FP with available match in esc camera: " + str(total_fp_av_esc))
    print("Total FP with not available match in esc camera: " + str(total_fp_not_av_esc))
    print("Total FN in esc camera: " + str(total_fn_esc))

    print("\nTotal TP in fac camera: " + str(total_tp_fac))
    print("Total TN in fac camera: " + str(total_tn_fac))
    print("Total FP with available match in fac camera: " + str(total_fp_av_fac))
    print("Total FP with not available match in fac camera: " + str(total_fp_not_av_fac))
    print("Total FN in fac camera: " + str(total_fn_fac))

    # Write paths of esc and fac images for facial recognition
    esc_file = ROOT + "esc_facial_paths.txt"
    with open(esc_file, 'w+') as file:
        for clip_paths in esc_facial_paths:
            file.write(str(clip_paths) + "\n")

    fac_file = ROOT + "fac_facial_paths.txt"
    with open(fac_file, 'w+') as file:
        for clip_paths in fac_facial_paths:
            file.write(str(clip_paths) + "\n")

    esc_FP_file = ROOT + "esc_FP_facial_paths.txt"
    with open(esc_FP_file, 'w+') as file:
        for clip_paths in esc_FP_facial_paths:
            file.write(str(clip_paths) + "\n")

    fac_FP_file = ROOT + "fac_FP_facial_paths.txt"
    with open(fac_FP_file, 'w+') as file:
        for clip_paths in fac_FP_facial_paths:
            file.write(str(clip_paths) + "\n")

if __name__ == '__main__':
    main()

