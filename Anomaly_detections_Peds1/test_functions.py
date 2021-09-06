# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Functions executed only in test
"""

from utils import *

#-----------------------------------------------------------------------------#
#Snap test trajectories saving their id
#-----------------------------------------------------------------------------#
def snap_test (trajectories, grid, cell_size, file_preffix):

    #Empty list of snapped trajectories
    snapped_trajectories = []

    #For each trajectory
    for trajectory in trajectories:

        #Initialize last cell's id and current cell's id
        last_point_cell = 1000000
        cell_id = 0

        #Empty list for each snapped trajectory
        snapped_trajectory = []
        #Append the id to the snapped trajectory
        snapped_trajectory.append(trajectory[0])

        #For each point in the original trajectory
        for point in trajectory[1]:

            #Look for the cell the point belongs to
            for cell in grid:

                if (cell[1][0] - cell_size[0]/2 < point[0] < cell[1][0] + cell_size[0]/2):
                    if (cell[1][1] - cell_size[1]/2 < point[1] < cell[1][1] + cell_size[1]/2):

                        #Store current cell's id
                        cell_id = cell[0]
                        break

                    #Points under the down limit of the grid
                    elif (point[1] > ((cell_size[1] * NUM_CELLS_VER) - 1) and
                          cell[0] in [c for c in range(NUM_CELLS_VER*NUM_CELLS_HOR - NUM_CELLS_HOR, NUM_CELLS_VER*NUM_CELLS_HOR)]):

                        #Store current cell's id
                        cell_id = cell[0]
                        break

            #If the last point of the trajectory was in the same cell,
            #update the timestamp of the last consecutive point in that cell
            if (last_point_cell == cell_id):

                last_point_timestamp = point[2]
                snapped_trajectory[-1][-1] = last_point_timestamp

            #If the last point was in another cell, create a new snapped point
            #and save it in the list of the current snapped trajectory
            else:

                first_point_timestamp = point[2]
                last_point_timestamp = point[2]
                snapped_point = [cell_id, grid[cell_id][1][0], grid[cell_id][1][1],
                                 first_point_timestamp, last_point_timestamp]
                snapped_trajectory.append(snapped_point)

                #Update last cell's id
                last_point_cell = cell_id

        #Append the information of the snapped trajectory to the global list
        snapped_trajectories.append(snapped_trajectory)

    #Write snapped trajectories in a file
    file = file_preffix + "_snapped_trajectories.txt"
    with open(file, 'w+') as file:
        for snapped_tra in snapped_trajectories:
            file.write("%s\n" % snapped_tra)

    return snapped_trajectories

#-----------------------------------------------------------------------------#
#Write zone id, cells, speeds and directions of test trajectories in a file
#-----------------------------------------------------------------------------#
def write_complete_test (file_preffix, complete_test_trajectories):

    file = file_preffix + "_complete_test.txt"
    with open(file, 'w+') as file:
        for zone in complete_test_trajectories:
            for z in zone:
                file.write("*" + str(z) + "\n")
            file.write("\n")
