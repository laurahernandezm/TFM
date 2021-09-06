# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Functions executed only in training
"""

from utils import *

#-----------------------------------------------------------------------------#
#Snap train trajectories ignoring their id
#-----------------------------------------------------------------------------#
def snap_train (trajectories, cells_summary, grid, cell_size, train_snapped_trajectories):

    #Empty list of snapped trajectories
    snapped_trajectories = []

    #For each trajectory
    for trajectory in trajectories:

        #Initialize last cell's id and current cell's id
        last_point_cell = 1000000
        cell_id = 0

        #Empty list for each snapped trajectory
        snapped_trajectory = []

        #For each point in the original trajectory
        for point in trajectory[1]:

            #Look for the cell the point belongs to
            for cell in grid:

                if (cell[1][0] - cell_size[0]/2 < point[0] < cell[1][0] + cell_size[0]/2):
                    if (cell[1][1] - cell_size[1]/2 < point[1] < cell[1][1] + cell_size[1]/2):

                        #Store current cell's id
                        cell_id = cell[0]
                        #Increment trajectory points in the cell
                        cells_summary[cell_id][0] += 1
                        break

                    #Points under the down limit of the grid
                    elif (point[1] > ((cell_size[1] * NUM_CELLS_VER) - 1) and
                          cell[0] in [c for c in range(NUM_CELLS_VER*NUM_CELLS_HOR - NUM_CELLS_HOR, NUM_CELLS_VER*NUM_CELLS_HOR)]):

                        #Store current cell's id
                        cell_id = cell[0]
                        #Increment trajectory points in the cell
                        cells_summary[cell_id][0] += 1
                        break

            #If the last point of the trajectory was in the same cell,
            #update the timestamp of the last consecutive point in that cell
            if (last_point_cell == cell_id):

                last_point_timestamp = point[2]
                snapped_trajectory[-1][-4] = last_point_timestamp
                #Distance
                snapped_trajectory[-1][-2] += distance.euclidean ([point[0], point[1]],
                                                                  snapped_trajectory[-1][-3])
                #Last point
                snapped_trajectory[-1][-3] = [point[0], point[1]]
                #Speed
                snapped_trajectory[-1][-1] = (snapped_trajectory[-1][-2] /
                                              (float(last_point_timestamp) - float(snapped_trajectory[-1][-5]) + 1))
            #If the last point was in another cell, create a new snapped point
            #and save it in the list of the current snapped trajectory
            else:

                first_point_timestamp = point[2]
                last_point_timestamp = point[2]
                last_point = [point[0], point[1]]
                snapped_point = [cell_id, grid[cell_id][1][0], grid[cell_id][1][1],
                                 first_point_timestamp, last_point_timestamp, last_point, 0, 0]
                snapped_trajectory.append(snapped_point)

                #Update last cell's id
                last_point_cell = cell_id

        #Append the information of the snapped trajectory to the global list
        snapped_trajectories.append(snapped_trajectory)

    #Accumulate all train trajectories
    train_snapped_trajectories.extend(snapped_trajectories)

    return cells_summary, train_snapped_trajectories

#-----------------------------------------------------------------------------#
#Calculate the average speed in each cell and clean snapped trajectories
#-----------------------------------------------------------------------------#
def cell_average_speed (cells_to_filter, snapped_trajectories, preffix):

    #Filter cells with at least one point
    cells_summary = {

        k: v
        for k, v in cells_to_filter.items()
        if v[0] != 0}

    #For each cell with at least one point mapped to its centroid
    for cell_id, info in cells_summary.items():

        accum_speed = 0
        num_snapped_points_cell = 0

        #Look for the snapped points in the cell
        for snapped_trajectory in snapped_trajectories:

            for snapped_point in snapped_trajectory:

                if (snapped_point[0] == cell_id):

                    #Increment the number of snapped points in the cell
                    num_snapped_points_cell += 1
                    #Accumulate speed
                    accum_speed += 0 if snapped_point[-1] == 0 else 1 / snapped_point[-1]
                    #Clean snapped point
                    del snapped_point[-3:]

        #Calculate average speed in the cell
        average_speed = 0 if accum_speed == 0 else num_snapped_points_cell / accum_speed

        #Store average speed in cells' summary
        info[-1] = average_speed

    #Write cells' summary in a file
    file = preffix + "_cells_summary.txt"
    with open(file, 'w+') as file:
        for cell_id, info in cells_summary.items():
            file.write(str(cell_id) + ": " + str(info) + "\n")

    return cells_summary, snapped_trajectories

#-----------------------------------------------------------------------------#
#Write all snapped trajectories from train in a file
#-----------------------------------------------------------------------------#
def save_train_snapped_trajectories (train_snapped_trajectories):

    file = train_result_folder + "train_snapped_trajectories.txt"
    with open(file, 'w+') as file:
        for snapped_tra in train_snapped_trajectories:
            file.write("%s\n" % snapped_tra)

#-----------------------------------------------------------------------------#
#Clustering
#-----------------------------------------------------------------------------#
def clustering (cells_summary, file_preffix):

    #Turn dictionary into matrix
    cells_summary_matrix = []

    for cell, info in cells_summary.items():
        cells_summary_matrix.append(info)

    #Scale matrix values
    cells_summary_matrix_scaled = scale(cells_summary_matrix, axis = 1)

    #Weight for each column
    weights = [1, 5, 5, 3]

    for cell in cells_summary_matrix_scaled:
        for i in range (0, 4):
            cell[i] *= weights[i]

    #Apply affinity propagation clustering
    #(To work on CCTV dataset change preference = -450 to -300) NOT AVAILABLE
    clusters = AffinityPropagation(damping = 0.9, preference = -450, max_iter = 1000, random_state = 123).fit(cells_summary_matrix_scaled)

    #Store the id of each cell
    cells_ids = []
    for k, v in cells_summary.items():
        cells_ids.append(k)

    #List to save the connected zones of each cluster
    zones = []

    #Dictionary to store the position of the list where a zone starts
    added_zones = {k: [] for k in range (0, max(clusters.labels_) + 1)}
    removed_indices = []

    last_label = -1
    last_zone_index = -1

    #For each cell, append it to a zone
    for i in range (0, len(cells_summary_matrix_scaled)):

        #If the current cell has the same clustering label as the previous, they
        #are in the same zone (neighbours)
        if (clusters.labels_[i] == last_label and cells_ids[i] == cells_ids[i-1] + 1):
            zones[last_zone_index].append(cells_ids[i])

            #If there is more than one zone for that cluster, check if they can
            #fuse in one
            if (len(added_zones[clusters.labels_[i]]) > 1):
                added = False

                for possible_zone in added_zones[clusters.labels_[i]]:
                    if (not added and possible_zone != last_zone_index):
                        for cell_j in zones[possible_zone]:
                            #If two cells are neighbours
                            if (cell_j == cells_ids[i]-NUM_CELLS_HOR or cell_j == cells_ids[i]+NUM_CELLS_HOR or
                                cell_j == cells_ids[i]-1 or cell_j == cells_ids[i]+1 or
                                cell_j == cells_ids[i]+NUM_CELLS_HOR-1 or cell_j == cells_ids[i]-NUM_CELLS_HOR+1 or
                                cell_j == cells_ids[i]+NUM_CELLS_HOR+1 or cell_j == cells_ids[i]-NUM_CELLS_HOR-1):

                                #Fuse lists
                                zones[possible_zone] = zones[possible_zone] + zones[last_zone_index]
                                #Remove the index of the second list
                                added_zones[clusters.labels_[i]].remove(last_zone_index)
                                removed_indices.append(last_zone_index)
                                last_zone_index = possible_zone
                                added = True
                                break

        #If the current cell is not from the same cluster
        else:
            #If the cell is the first of its cluster, append it to the list and
            #save the position of the list where it has been added
            if (added_zones[clusters.labels_[i]] == []):

                zone = [cells_ids[i]]
                zones.append(zone)
                added_zones.update({clusters.labels_[i]: added_zones[clusters.labels_[i]] +
                                    [len(zones) - 1]})
                last_zone_index = len(zones) - 1

            #If the cell is not the first of its cluster
            else:
                added = False
                #Check possible zones where it can be added
                for possible_zone in added_zones[clusters.labels_[i]]:
                    if (not added):
                        for cell_j in zones[possible_zone]:

                            #If the current cell is neighbour of any cell in the zone, append it
                            if (cell_j == cells_ids[i]-NUM_CELLS_HOR or cell_j == cells_ids[i]+NUM_CELLS_HOR or
                                cell_j == cells_ids[i]-1 or cell_j == cells_ids[i]+1 or
                                cell_j == cells_ids[i]+NUM_CELLS_HOR-1 or cell_j == cells_ids[i]-NUM_CELLS_HOR+1 or
                                cell_j == cells_ids[i]+NUM_CELLS_HOR+1 or cell_j == cells_ids[i]-NUM_CELLS_HOR-1):

                                zones[possible_zone].append(cells_ids[i])
                                last_zone_index = possible_zone
                                added = True
                                break
                #If the cell has not got neighbours in already created zones,
                #create a new one and store start index
                if (not added):

                    zone = [cells_ids[i]]
                    zones.append(zone)
                    added_zones.update({clusters.labels_[i]: added_zones[clusters.labels_[i]] +
                                        [len(zones) - 1]})
                    last_zone_index = len(zones) - 1

            last_label = clusters.labels_[i]

    #If there is more than one zone for that cluster, check again if they can
    #fuse in one
    for i in range(0, max(clusters.labels_) + 1):
        if (len(added_zones[i]) > 1):
            ignore = []
            possible_indices = added_zones[i]

            for j in range(len(possible_indices)):
                if (j not in ignore):
                    for k in range(len(possible_indices)):

                        if (j != k and k not in ignore):

                            zone_a = zones[possible_indices[j]]
                            zone_b = zones[possible_indices[k]]

                            if (any(cell_a in zone_b for cell_a in [a-NUM_CELLS_HOR-1 for a in zone_a]) or
                                any(cell_a in zone_b for cell_a in [a-NUM_CELLS_HOR for a in zone_a]) or
                                any(cell_a in zone_b for cell_a in [a-NUM_CELLS_HOR+1 for a in zone_a]) or
                                any(cell_a in zone_b for cell_a in [a-1 for a in zone_a]) or
                                any(cell_a in zone_b for cell_a in [a+1 for a in zone_a]) or
                                any(cell_a in zone_b for cell_a in [a+NUM_CELLS_HOR-1 for a in zone_a]) or
                                any(cell_a in zone_b for cell_a in [a+NUM_CELLS_HOR for a in zone_a]) or
                                any(cell_a in zone_b for cell_a in [a+NUM_CELLS_HOR+1 for a in zone_a])):

                                #Fuse lists
                                zones[possible_indices[j]] = zones[possible_indices[j]] + zones[possible_indices[k]]
                                #Remove the index of the second list
                                ignore.append(possible_indices[k])
                                removed_indices.append(possible_indices[k])

    #Remove deleted zones
    zones_def = []
    for zone, cells in added_zones.items():
        for index in cells:
            zones_def.append([zone, zones[index]])

    #Write zones in a file
    file = file_preffix + "_zones.txt"
    with open(file, 'w+') as file:
        for zone in zones_def:
            file.write(str(zone) + "\n")

    return clusters, zones_def
