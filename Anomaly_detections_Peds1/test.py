# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Anomaly detection on test videos (according to the last train information)
"""

from test_functions import *
from zones import *
from anomaly_detections import *

def main():

    print("----Testing on " + tracking_results_test + " data----")

    #Make the grid
    grid, cell_size = make_grid()

    #Get frame limits
    x_min, y_min, x_max, y_max = bounds(grid, cell_size)

    #Read zones information from training
    zones_with_default = []

    file_tzd = train_result_folder + "train_zones_default.txt"
    zones_w_def = open(file_tzd, 'r')

    #Read all lines from the file
    zs_info = zones_w_def.readlines()
    zones_w_def.close()

    #Save the information as a list
    for index, info in enumerate(zs_info):
        #Start new zone
        if (info == "\n" or index == 0):
            zones_with_default.append([])

        #Append info to the last started zone
        if (info != "\n"):
            zones_with_default[-1].append(literal_eval(info[1:-1]))

    del zones_with_default[-1]

    #Read zones information (with speeds and directions) from training
    zones_main_direction_default = []

    file_tmdd = train_result_folder + "train_main_direction_default.txt"
    zones_main_dir_def = open(file_tmdd, 'r')

    #Read all lines from the file
    zs_info = zones_main_dir_def.readlines()
    zones_main_dir_def.close()

    #Save the information as a list
    for index, info in enumerate(zs_info):
        #Start new zone
        if (info == "\n" or index == 0):
            zones_main_direction_default.append([])

        #Append info to the last started zone
        if (info != "\n"):
            zones_main_direction_default[-1].append(literal_eval(info[1:-1]))

    del zones_main_direction_default[-1]

    #-----------------------------------------------------------------------------#
    # TEST
    #-----------------------------------------------------------------------------#
    print("Processing test videos ...")

    #Get test trajectories to detect anomalies
    for result in os.listdir (tracking_results_test):
        if os.path.isfile(os.path.join(tracking_results_test, result)) and result.endswith(".txt"):

            #Preffix for each video
            #(To work on CCTV dataset change [:-4] to [:-11]) NOT AVAILABLE
            file_preffix = test_result_folder + result[:-4]

            #Input trajectories
            result_path = tracking_results_test + "\\" + result
            test_trajectories = extract_trajectories (result_path, True)

            #Snap extracted trajectories in the grid
            test_snapped_trajectories = snap_test (test_trajectories, grid, cell_size, file_preffix)

            #Append snapped points to each zone
            test_trajectories_in_zones = trajectories_in_zones (zones_with_default, test_snapped_trajectories,
                                                                file_preffix, True)
            #Append speed to each trajectory
            test_trajectories_with_speed = speed_in_zones (test_trajectories_in_zones,
                                                           file_preffix, True)
            #Append direction to each trajectory
            complete_test_trajectories = direction_in_zones (test_trajectories_with_speed,
                                                        file_preffix, True)

            #In test we do not need zone information, only trajectories
            for zone in complete_test_trajectories:
                del zone[2]

            #Write test information to a file
            write_complete_test (file_preffix, complete_test_trajectories)

            #Detect abnormal speeds
            zones_with_abnormal_speeds = abnormal_speeds (zones_main_direction_default,
                                                          file_preffix, complete_test_trajectories)
            #Detect abnormal directions
            zones_with_abnormal_directions = abnormal_directions (zones_with_abnormal_speeds,
                                                                  file_preffix, complete_test_trajectories)

            anomalies = deepcopy (zones_with_abnormal_directions)

            #To draw bounding boxes of abnormal individuals, we only need
            #the abnormal trajectories in each zone. After remove these
            #information we will have: zone id, main directions, abnormal
            #speeds and abnormal directions
            for zone in anomalies:
                del zone[6]
                del zone[5]
                del zone[4]
                del zone[3]
                del zone[2]
                del zone[1]

            #Filter to show only trajectories with abnormal speed
            #filter_trajectories_speed (result_path, anomalies, file_preffix)

            #Filter to show only trajectories with abnormal direction
            #filter_trajectories_direction (result_path, anomalies, file_preffix)

            #Filter to show only abnormal trajectories
            filter_trajectories_both (result_path, anomalies, file_preffix, x_min, y_min, x_max, y_max)

if __name__ == '__main__':
    main()
