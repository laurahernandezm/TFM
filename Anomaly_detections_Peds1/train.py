# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Train to obtain the zones and their information
"""

from training_functions import *
from zones import *

def main():

    print("----Training on " + tracking_results_train + " data----")

    #Create folders to store information
    create_folders()

    #Make the grid
    grid, cell_size = make_grid()

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

if __name__ == '__main__':
    main()
