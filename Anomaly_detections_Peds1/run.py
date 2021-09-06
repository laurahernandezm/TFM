# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Complete execution
"""

from training_functions import *
from test_functions import *
from zones import *
from anomaly_detections import *

def main():

    print("----Running algorithm (train + test)----")

    #Create folders to store information
    create_folders()

    #Make the grid
    grid, cell_size = make_grid()

    #Get frame limits
    x_min, y_min, x_max, y_max = bounds(grid, cell_size)

    #-----------------------------------------------------------------------------#
    # TRAIN
    #-----------------------------------------------------------------------------#
    #Initialize each cell with 0 trajectory points inside
    cells_summary_complete = initialize_cells_summary (grid)
    cells_summary = initialize_cells_summary (grid)

    #List to get training trajectories
    train_snapped_trajectories = []

    print("Processing training videos ...")

    for result in os.listdir (tracking_results_train):
        if os.path.isfile(os.path.join(tracking_results_train, result)) and result.endswith(".txt"):

            #Input trajectories
            result_path = tracking_results_train + "\\" + result
            trajectories = extract_trajectories (result_path, True)

            #Snap extracted trajectories in the grid
            cells_summary, train_snapped_trajectories = snap_train (trajectories, cells_summary, grid, cell_size, train_snapped_trajectories)

    file_preffix = train_result_folder + "train"

    #Calculate cells' average speed and clean snapped trajectories removing
    #additional information added to snapped points (last point, distance and speed)
    cells_summary, train_snapped_trajectories = cell_average_speed (cells_summary,
                                                             train_snapped_trajectories, file_preffix)

    #Save all train snapped trajectories
    save_train_snapped_trajectories (train_snapped_trajectories)

    #Apply clustering
    clusters, zones = clustering (cells_summary, file_preffix)

    #Append snapped points to each zone
    zones_with_trajectories = trajectories_in_zones (zones, train_snapped_trajectories,
                                                    file_preffix, False)
    #Append speeds to each zone
    zones_with_speeds = speed_in_zones (zones_with_trajectories, file_preffix, False)

    #Append directions to each zone
    zones_with_directions = direction_in_zones (zones_with_speeds, file_preffix, False)

    #Append mean and standard deviation to each zone
    zones_with_gaussian = gaussian_dist (zones_with_directions, file_preffix)

    #Detect main direction in each zone
    zones_main_direction = main_direction (zones_with_gaussian, file_preffix)

    #Save frequency histograms for the directions inside each zone
    histograms (zones_with_directions, file_preffix)

    #Add a default zone
    zones_with_default = default_zone (zones_with_trajectories)

    zones_main_direction_default = default_zone_sp_dir (zones_main_direction,
                                                        zones_with_default)
    #Transform cells id into their centroid
    zones_centr = transform_zones (zones_main_direction_default, cells_summary_complete)

    #Display image with the grid (each cell's centroid in white), clusters and directions of the trajectories
    grided_trajectories (trajectories, grid, cell_size,
                         train_snapped_trajectories, cells_summary, clusters,
                         zones_centr, zones_main_direction_default, file_preffix)

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
