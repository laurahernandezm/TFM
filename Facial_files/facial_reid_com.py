# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Send images from ReID module to the facial recognition system
"""

import cv2
import zmq
import base64
import struct
import glob
import os

available_identities = ['alberto', 'carlos', 'german', 'jesus', 'jose', 'juanlu', 'laura', 'nacho', 'nuria', 'paco']

# Mean scores lesser than this value are discarded
SCORE_THRESHOLD = 0.2

# Possible voting schemes: sum (sum scores from each identity and choose the
# higher result) and count (choose the most repeated)
VOTING_TYPE = 'sum'

# Take into account the best K predictions (None to take all)
NUM_PREDICTIONS = None

# Folder with the query images
IMAGES_FOLDER = 'facial_imgs_FP'

#------------------------------------------------------------------------------#
# Send an image through the socket
#------------------------------------------------------------------------------#
def send_image_through_socket(img, socket):
    dim_x, dim_y = img.shape[:2]
    encoded_img = base64.b64encode(cv2.imencode(".png", img)[1])
    socket.send_multipart([str(dim_x).encode(), str(dim_y).encode(), encoded_img])

#------------------------------------------------------------------------------#
# Get predictions from the socket
#------------------------------------------------------------------------------#
def receive_predictions(socket):
    message = socket.recv_multipart()
    status, result = message[0].decode('utf-8'), message[1:]
    predictions = []
    if status == "SUCCESS":
        for bin_name, bin_score in zip(result[0::2], result[1::2]):
            # Binary name is converted to utf-8
            # Score is changed from big-endian to little-endian
            # (bytestring reversed) and decoded
            predictions.append(
                (bin_name.decode('utf-8'),
                 struct.unpack("f", bin_score[::-1])[0])
            )

        return status, predictions

    return status, result

#------------------------------------------------------------------------------#
# Compute results
# Not available == the identity is not registered
# Weak prediction == there is a match, but its score is lower than the threshold
# Match == the identity is registered and the match is correct
# Fail == the identity we are looking for is available, but the match is incorrect
#------------------------------------------------------------------------------#
def check_current_match(img_path, predictions, n_preds = None, score_thr = 0.2, voting_type = 'sum'):

    img_id = os.path.splitext(os.path.basename(img_path))[0]
    score_matches = {}
    count_matches = {}

    if n_preds is not None:
        predictions = predictions[:n_preds]
    for filename, score in predictions:
        pred_id = filename.split("/")[0]
        if pred_id in score_matches.keys():
            score_matches[pred_id] += score
            count_matches[pred_id] += 1
        else:
            score_matches[pred_id] = score
            count_matches[pred_id] = 1

    matches = {}
    if voting_type == 'sum':
        matches = score_matches
    elif voting_type == 'count':
        matches = count_matches
    else:
        raise NotImplementedError(
            f"{voting_type} voting scheme is not currently supported"
        )

    max_id = max(matches.items(), key=lambda x: x[1])[0]

    if img_id not in available_identities:
        return 'not_available'
    elif score_matches[max_id] / count_matches[max_id] <= score_thr:
        return 'weak_prediction'
    elif max_id == img_id:
        return 'match'
    else:
        return 'fail'

#------------------------------------------------------------------------------#
# Connect with the system, send images to it and check if matches are correct
#------------------------------------------------------------------------------#
def main():

    # Connection to the system
    endpoint = "tcp://fluque.ugr.es:4242"

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)

    socket.connect(endpoint)

    # Get images from fac camera
    fac_images = glob.glob(f"{IMAGES_FOLDER}/*/fac/*")

    non_faces = 0
    corrects = 0
    unavailable = 0
    incorrects = 0
    weak_predictions = 0

    for img_path in fac_images:

        img = cv2.imread(img_path)
        send_image_through_socket(img, socket)
        status, result = receive_predictions(socket)

        if status == "SUCCESS":
            match_kind = check_current_match(
                img_path,
                result,
                n_preds=NUM_PREDICTIONS,
                score_thr=SCORE_THRESHOLD,
                voting_type=VOTING_TYPE
            )

            if match_kind == 'not_available':
                unavailable += 1
            elif match_kind == 'match':
                corrects += 1
            elif match_kind == 'fail':
                incorrects += 1
            elif match_kind == 'weak_prediction':
                weak_predictions += 1
        else:
            non_faces += 1

    print("RESULTS IN FACIAL CAMERA:")
    print(f'Total number of images: {len(fac_images)}')
    print(f'Number of images without faces: {non_faces}')
    print(f'Number of images to evaluate: {len(fac_images) - non_faces}')
    print(f'Number of correct matches: {corrects} ({100*corrects/(len(fac_images) - non_faces)} % of images with faces)')
    print(f'Number of incorrect matches: {incorrects} ({100*incorrects/(len(fac_images) - non_faces)} % of images with faces)')
    print(f'Number of weak predictions: {weak_predictions} ({100*weak_predictions/(len(fac_images) - non_faces)} % of images with faces)')
    print(f'Number of not available identities: {unavailable} ({100*unavailable/(len(fac_images) - non_faces)} % of images with faces)')

    # Get images from esc camera
    esc_images = glob.glob(f"{IMAGES_FOLDER}/*/esc/*")

    non_faces = 0
    corrects = 0
    unavailable = 0
    incorrects = 0
    weak_predictions = 0

    for img_path in esc_images:

        img = cv2.imread(img_path)
        send_image_through_socket(img, socket)
        status, result = receive_predictions(socket)

        if status == "SUCCESS":
            match_kind = check_current_match(
                img_path,
                result,
                n_preds=NUM_PREDICTIONS,
                score_thr=SCORE_THRESHOLD,
                voting_type=VOTING_TYPE
            )

            if match_kind == 'not_available':
                unavailable += 1
            elif match_kind == 'match':
                corrects += 1
            elif match_kind == 'fail':
                incorrects += 1
            elif match_kind == 'weak_prediction':
                weak_predictions += 1
        else:
            non_faces += 1

    print("RESULTS IN LADDERS CAMERA:")
    print(f'Total number of images: {len(esc_images)}')
    print(f'Number of images without faces: {non_faces}')
    print(f'Number of images to evaluate: {len(esc_images) - non_faces}')
    print(f'Number of correct matches: {corrects} ({100*corrects/(len(esc_images) - non_faces)} % of images with faces)')
    print(f'Number of incorrect matches: {incorrects} ({100*incorrects/(len(esc_images) - non_faces)} % of images with faces)')
    print(f'Number of weak predictions: {weak_predictions} ({100*weak_predictions/(len(esc_images) - non_faces)} % of images with faces)')
    print(f'Number of not available identities: {unavailable} ({100*unavailable/(len(esc_images) - non_faces)} % of images with faces)')

if __name__ == '__main__':
    main()
