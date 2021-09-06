# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Utils
"""

import cv2
import numpy
from scipy.spatial import distance
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from matplotlib import colors as my_colors
import math
import os
from os import path as osp
from copy import deepcopy
from cycler import cycler as cy
from collections import defaultdict, Counter
from datasets import Datasets
import glob
from natsort import natsorted
from ast import literal_eval

#Grid of NUM_CELLS x NUM_CELLS
NUM_CELLS_HOR = 64
NUM_CELLS_VER = 54

#Coefficient of sensitivity
ALPHA = 0.9

#Factors to refine speed confidence intervals
#(To work on CCTV dataset change LOW_FACTOR to 1.9 and HIGH_FACTOR to 1.6) NOT AVAILABLE
LOW_FACTOR = 1.3
HIGH_FACTOR = 1.1

#Number of possible directions
NUM_DIRECTIONS = 9

#Compound directions
COMPOUND_DIRECTIONS = {"N": ["NE", "NW"], "NE": ["N", "E"], "NW": ["N", "W"],
                       "S": ["SE", "SW"], "SE": ["S", "E"], "SW": ["S", "W"],
                       "NNE": ["N", "NE"], "ENE": ["NE", "E"],
                       "NNW": ["N", "NW"], "WNW": ["NW", "W"],
                       "SSE": ["S", "SE"], "ESE": ["SE", "E"],
                       "SSW": ["S", "SW"], "WSW": ["SW", "W"],
                       "E": ["NE", "SE"], "W": ["NW", "SW"], "X": []}

#Opposite directions
OPPOSITE_DIRECTIONS = {"N": ["S", "E", "W", "SE", "SW", "ENE", "ESE", "SSE", "SSW", "WSW", "WNW"],
                       "NE": ["S", "W", "SE", "SW", "NW", "ESE", "SSE", "SSW", "WSW", "WNW", "NNW"],
                       "NW": ["S", "E", "NE", "SE", "SW", "NNE", "ENE", "ESE", "SSE", "SSW", "WSW"],
                       "S": ["N", "E", "W", "NE", "NW", "NNE", "ENE", "ESE", "WSW", "WNW", "NNW"],
                       "SE": ["N", "W", "NE", "SW", "NW", "NNE", "ENE", "SSW", "WSW", "WNW", "NNW"],
                       "SW": ["N", "E", "NE", "SE", "NW", "NNE", "ENE", "ESE", "SSE", "WNW", "NNW"],
                       "E": ["N", "S", "W", "SW", "NW", "NNE", "SSE", "SSW", "WSW", "WNW", "NNW"],
                       "W": ["N", "S", "E", "NE", "SE", "NNE", "ENE", "ESE", "SSE", "SSW", "NNW"],
                       "NNE": ["S", "E", "W", "SE", "SW", "NW", "ESE", "SSE", "SSW", "WSW", "WNW"],
                       "ENE": ["N", "S", "W", "SE", "SW", "NW", "SSE", "SSW", "WSW", "WNW", "NNW"],
                       "NNW": ["S", "E", "W", "SE", "SW", "NE", "ENE", "ESE", "SSE", "SSW", "WSW"],
                       "WNW": ["N", "S", "E", "NE", "SE", "SW", "NNE", "ENE", "ESE", "SSE", "SSW"],
                       "SSE": ["N", "E", "W", "NE", "SW", "NW", "NNE", "ENE", "WSW", "WNW", "NNW"],
                       "ESE": ["N", "S", "W", "NE", "SW", "NW", "NNE", "SSW", "WSW", "WNW", "NNW"],
                       "SSW": ["N", "E", "W", "NE", "SE", "NW", "NNE", "ENE", "ESE", "WNW", "NNW"],
                       "WSW": ["N", "S", "E", "NE", "SE", "NW", "NNE", "ENE", "ESE", "SSE", "NNW"],
                       "X": [],
                       "Def": ["N", "S", "E", "W", "NE", "SE", "NW", "SW", "NNE",
                               "ENE", "ESE", "SSE", "NNW", "WNW", "WSW", "SSW", "X"]}

#Colors for the clusters
cv2_colors = []

for color in list(my_colors.CSS4_COLORS.keys())[30:]:
    hex_color = my_colors.to_hex(color)
    rgb_color = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    cv2_colors.append(bgr_color)

#Input image
img = cv2.imread("./empty_scene.jpg")

#Folders with information about tracked videos
tracking_results_train = "./tracking_results/train/"
tracking_results_test = "./tracking_results/test/"

#Folder to store informative files
result_folder = "./information/"

#Folder to store informative files from train
train_result_folder = "./information/train/"

#Folder to store informative files from test
test_result_folder = "./information/test/"
#-----------------------------------------------------------------------------#
#Create folders to store informative files if they don't exist
#-----------------------------------------------------------------------------#
def create_folders ():

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    if not os.path.exists(train_result_folder):
        os.makedirs(train_result_folder)

    if not os.path.exists(test_result_folder):
        os.makedirs(test_result_folder)

#-----------------------------------------------------------------------------#
#Store id and coordinates of the centroid of each cell of a grid of NUM_CELLS x NUM_CELLS
#-----------------------------------------------------------------------------#
def make_grid ():

    #Image size
    height, width, channels = img.shape

    #Size of each cell
    hor_grid_size = int(width / NUM_CELLS_HOR)
    ver_grid_size = int(height / NUM_CELLS_VER)
    cell_size = (hor_grid_size, ver_grid_size)

    #List to store cells
    grid = []
    cell_id = 0

    #Calculate the centroid of each cell and save it with its id
    for y in range(0, height - 1, ver_grid_size):

        #Centroid y-coordinate
        y_centr = int(y + ver_grid_size/2)

        for x in range(0, width - 1, hor_grid_size):

            #Centroid x-coordinate
            x_centr = int(x + hor_grid_size/2)

            #Save cell information
            cell = []
            cell.append(cell_id)
            centroid = (x_centr, y_centr)
            cell.append(centroid)

            #Append the cell to the grid
            grid.append(cell)

            #Increment cell id
            cell_id = cell_id + 1

    return grid, cell_size

#-----------------------------------------------------------------------------#
#Calculate x and y max and min values for the frame size we are working with
#-----------------------------------------------------------------------------#
def bounds (grid, cell_size):

    # Left limit (first cell's leftmost x value)
    x_min = grid[0][1][0] - cell_size[0]/2
    # Upper limit (first cell's top y value)
    y_min = grid[0][1][1] - cell_size[1]/2
    # Right limit (last cell's rightmost x value)
    x_max = grid[-1][1][0] + cell_size[0]/2
    # Lower limit (last cell's bottom y value)
    y_max = grid[-1][1][1] + cell_size[1]/2

    return x_min, y_min, x_max, y_max

#-----------------------------------------------------------------------------#
#Convert input and create a list of trajectories with their id and points.
#Remove trajectories that last less than 20 frames
#-----------------------------------------------------------------------------#
def extract_trajectories (file, all_traj_in_frame):

    #Input file with MOT format (frame, id, bb_left, bb_top, bb_width, bb_height)
    file_input = open (file, "r")

    #Read all lines from input file
    trs = file_input.readlines()

    #Empty list for the trajectories
    trajectories = []

    #Indices of short trajectories in the list to delete them
    short_indices = []

    #Id of the last processed trajectory
    last_id = 0

    #Indices of existing trajectories (used when all_traj_in_frame = True)
    existing_tr = []

    #Dictionary to control the number of frames of each trajectory
    tr_frames = {}

    #For each line from the input file
    for tr in trs:

        #Ignore blank lines
        if (tr != "\n"):

            split_tr = tr.split(",")

            frame_id = split_tr[0]
            tr_id = split_tr[1]
            x_left = split_tr[2]
            y_top = split_tr[3]
            width = split_tr[4]
            height = split_tr[5]

            #Get the midpoint of the lower segment
            x_mid = float(x_left) + float(width)/2
            y_mid = float(y_top) + float(height)

            #If we have all the trajectories frame by frame, i.e first line is
            #frame 1, trajectory 1, second line is frame 1, trajectory 2...
            if (all_traj_in_frame):

                #If the current line belongs to an existing trajectory:
                if (tr_id in existing_tr):

                    #Add the new point to the trajectory
                    time = frame_id
                    mid_point = (x_mid, y_mid, time)
                    trajectories[int(tr_id) - 1][1].append(mid_point)
                    tr_frames[tr_id] += 1
                #If the current line does not belong to any trajectory:
                else:

                    #Create a new trajectory list with its id and empty point list
                    new_tr = []
                    new_tr.append(tr_id)
                    new_tr.append([])

                    #Add the new point to the trajectory. Each point is a tuple (x, y, time, width, height)
                    time = frame_id
                    mid_point = (x_mid, y_mid, time, width, height)
                    new_tr[1].append(mid_point)

                    #Add the new trajectory to the list of trajectories
                    if (int(tr_id) - 1 == last_id):
                        trajectories.append(new_tr)
                    # If current trajectory id is not consecutive
                    else:
                        for i in range(0, (int(tr_id) - 1) - int(last_id)):
                            trajectories.append("deleted")
                            tr_frames[(int(last_id) + i + 1)] = 0
                        trajectories.append(new_tr)

                    last_id = tr_id
                    existing_tr.append(tr_id)
                    tr_frames[tr_id] = 1

            #If we have all the frames for a trajectory, i.e first line is frame
            #1, trajectory 1, second line is frame 2, trajectory 1...
            else:
                #If the current line belongs to the same trajectory as the last:
                if (last_id == tr_id):

                    #Add the new point to the trajectory
                    time = frame_id
                    mid_point = (x_mid, y_mid, time)
                    trajectories[int(tr_id) - 1][1].append(mid_point)
                    tr_frames[tr_id] += 1
                #If the current line does not belong to the same trajectory as the last:
                else:

                    #Create a new trajectory list with its id and empty point list
                    new_tr = []
                    new_tr.append(tr_id)
                    new_tr.append([])

                    #Add the new point to the trajectory. Each point is a tuple (x, y, time, width, height)
                    time = frame_id
                    mid_point = (x_mid, y_mid, time, width, height)
                    new_tr[1].append(mid_point)

                    #Add the new trajectory to the list of trajectories
                    if (int(tr_id) - 1 == last_id):
                        trajectories.append(new_tr)
                    # If current trajectory id is not consecutive
                    else:
                        for i in range(0, (int(tr_id) - 1) - int(last_id)):
                            trajectories.append("deleted")
                            tr_frames[(int(last_id) + i + 1)] = 0
                        trajectories.append(new_tr)

                    last_id = tr_id
                    tr_frames[tr_id] = 1

    file_input.close()

    #If the trajectory lasts less than 20 frames it is added to the
    #short_indices list
    for trajectory_id, frame_count in tr_frames.items():
        if (frame_count < 20):
            short_indices.append(int(trajectory_id) - 1)

    #Delete trajectories that last less than 20 frames (1 second)
    short_indices.sort(reverse = True)

    for i in short_indices:
        del trajectories[i]

    return trajectories

#-----------------------------------------------------------------------------#
#Initialize cells summary with 0 points inside each, its centroid and speed = 0
#-----------------------------------------------------------------------------#
def initialize_cells_summary (grid):

    #Initialize cells summary
    num_points_inside = 0

    #Initialize each cell with 0 trajectory points inside
    cells_summary = {k: [num_points_inside] for k in range(grid[-1][0] + 1)}

    for cell in grid:
        #Centroid
        cells_summary[cell[0]].append(cell[1][0])
        cells_summary[cell[0]].append(cell[1][1])
        #Speed
        cells_summary[cell[0]].append(0)

    return cells_summary

#-----------------------------------------------------------------------------#
#Display trajectories in the grid
#-----------------------------------------------------------------------------#
def grided_trajectories (trajectories, grid, cell_size, snapped_trajectories,
                         cells_summary, clustering, zones, zones_traj, file_preffix):

    #Image size
    height, width, channels = img.shape

    #Draw centroids of each cell of the grid
    for c in grid:
        center = (int(c[1][0]), int(c[1][1]))
        cv2.circle(img, center, 1, (255, 255, 255), 1)

    #Draw vertical lines of the grid
    for x in numpy.arange(0, width - 1, cell_size[0]):
        cv2.line(img, (int(x), 0), (int(x), height), (255, 0, 0), 1, 1)

    #Draw horizontal lines of the grid
    for y in numpy.arange(0, height - 1, cell_size[1]):
        cv2.line(img, (0, int(y)), (width, int(y)), (255, 0, 0), 1, 1)

    '''
    #Draw trajectories
    for tra in trajectories:
        for point in tra[1]:

            coordinates = (int(point[0]), int(point[1]))
            cv2.circle(img, coordinates, 1, (0, 166, 255), 1)

    #Draw centroids of the cells containing at least one snapped point
    for snapped_tra in snapped_trajectories:
        for snapped_point in snapped_tra[1:]:

            coordinates = (int(snapped_point[1]), int(snapped_point[2]))
            cv2.circle(img, coordinates, 3, (0, 255, 0), -1)

    #Draw centroids of the cells according to the cluster they belong to
    i = 0

    for cell_id, info in cells_summary.items():

        cv2.circle(img, (info[1], info[2]), 3, cv2_colors[clustering.labels_[i]], -1)
        i+=1
    '''
    #Draw clusters
    for zone in zones:
        if (zone[0] != -1):
            for i in zone[1]:
                cv2.rectangle(img, (int(i[0] - cell_size[0]/2), int(i[1] - cell_size[1]/2)),
                              (int(i[0] + cell_size[0]/2), int(i[1] + cell_size[1]/2)),
                              cv2_colors[zone[0]], -1)

    #Draw trajectories' direction
    for zone in zones_traj:
        for tra in zone[2]:

            cv2.line(img, (tra[1][1], tra[1][2]), (tra[len(tra) - 1][1],
                                                   tra[len(tra) - 1][2]), (0, 0, 0), 2)

    #Display grided image
    window_name = "grid"
    cv2.imshow(window_name, img)
    file = file_preffix + "_grided.png"
    cv2.imwrite(file, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

#-----------------------------------------------------------------------------#
#Frequency histograms for the directions inside the zones
#-----------------------------------------------------------------------------#
def histograms (zones, file_preffix):

    i = 0
    for zone in zones:

        count = {"N": 0, "NE": 0, "NW": 0, "S": 0, "SE": 0, "SW": 0, "E": 0, "W": 0, "X": 0}

        for direction in zone[4]:
            count[direction[-1]] += 1

        plt.bar(count.keys(), count.values())
        file = file_preffix + "_zones[" + str(i) + "]_histogram.png"
        plt.savefig(file)
        plt.clf()

        i += 1
