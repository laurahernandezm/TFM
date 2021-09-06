# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Functions that update zones information, executed in both train and test
"""

from utils import *

#-----------------------------------------------------------------------------#
#Change cells' id for their centroid
#-----------------------------------------------------------------------------#
def transform_zones (zones, cells_summary):

    zones_with_centroids = []

    for zone in zones:
        zone_w_centr = [zone[0], []]
        for i in zone[1]:
            centr = [cells_summary[i][1], cells_summary[i][2]]
            zone_w_centr[-1].append(centr)
        zones_with_centroids.append(zone_w_centr)

    return zones_with_centroids

#-----------------------------------------------------------------------------#
#Append snapped trajectories to the zones
#-----------------------------------------------------------------------------#
def trajectories_in_zones (zones, snapped_trajectories, file_preffix, test):

    #Copy the information already known about the zones
    zones_w_traj = deepcopy(zones)

    invalid = []

    #For each zone, check every snapped trajectory to store the snapped points
    #belonging to that zone
    for i in range(0, len(zones)):
        zones_w_traj[i].append([])
        for snapped_tra in snapped_trajectories:
            point = []
            #If we are in test videos we store trajectories id
            if (test):
                point.append(snapped_tra[0])
                for snapped_point in snapped_tra[1:]:
                    for cell in zones[i][1]:
                        if (snapped_point[0] == cell):
                            point.append(snapped_point)
            else:
                for snapped_point in snapped_tra:
                    for cell in zones[i][1]:
                        if (snapped_point[0] == cell):
                            point.append(snapped_point)

            #Append only if the trajectory is longer than one point (train)
            if (not test and len(point) > 1):
                zones_w_traj[i][-1].append(point)
            #If test, append if the trajectory is longer than one point or,
            #in that case, append if the snapped point is 20 frames long or more
            elif (test and len(point) == 2 and (int(point[1][-1]) - int(point[1][-2])) >= 20):
                zones_w_traj[i][-1].append(point)
            elif (test and len(point) > 2):
                zones_w_traj[i][-1].append(point)

        #If there are not trajectories in a zone we can remove it
        if (len(zones_w_traj[i][-1]) == 0):
            invalid.append(i)

    invalid.sort(reverse = True)

    #Delete invalid zones
    for i in invalid:
        del zones_w_traj[i]

    #Delete training trajectories
    if (test):
        for zone in zones_w_traj:
            del zone[2]

    #Write zones with the snapped points in a file
    file = file_preffix + "_zones_with_trajectories.txt"
    with open(file, 'w+') as file:
        for zone in zones_w_traj:
            file.write(str(zone) + "\n\n")

    return zones_w_traj

#-----------------------------------------------------------------------------#
#Append speed of the snapped trajectories to the zones
#-----------------------------------------------------------------------------#
def speed_in_zones (zones, file_preffix, test):

    #Copy the information already known about the zones
    zones_w_sp = deepcopy(zones)

    #For each zone, calculate the speed for each snapped trajectory in it
    for i in range(0, len(zones)):
        zones_w_sp[i].append([])
        for tra in zones[i][2]:
            speed = []
            sumatory = 0
            #If we are in test videos we store trajectories id
            if (test):
                speed.append(tra[0])
                for j in range (1, len(tra) - 1):
                    sumatory += distance.euclidean([tra[j+1][1], tra[j+1][2]],
                                                   [tra[j][1], tra[j][2]], [1.0, 1.8])
            else:
                for j in range (0, len(tra) - 1):
                    sumatory += distance.euclidean([tra[j+1][1], tra[j+1][2]],
                                                   [tra[j][1], tra[j][2]], [1.0, 1.8])

            dif = int(tra[len(tra)-1][-1]) - int(tra[1][-2])

            speed.append(sumatory / 1 if (dif) == 0 else sumatory / dif)
            zones_w_sp[i][-1].append(speed)

    #Write zones with the speeds in a file
    file = file_preffix + "_zones_with_speeds.txt"
    with open(file, 'w+') as file:
        for zone in zones_w_sp:
            file.write(str(zone) + "\n\n")

    return zones_w_sp

#-----------------------------------------------------------------------------#
#Append direction of the snapped trajectories to the zones
#-----------------------------------------------------------------------------#
def direction_in_zones (zones, file_preffix, test):

    #Copy the information already known about the zones
    zones_w_dir = deepcopy(zones)

    #For each trajectory, calculate the overall direction between the last and
    #first snapped point
    for i in range(0, len(zones)):
        zones_w_dir[i].append([])
        for tra in zones[i][2]:
            direction = []
            #If we are in test videos we store trajectories id
            if (test):
                direction.append(tra[0])
                slope = tra[len(tra) - 1][2] - tra[1][2]
                dif_x = tra[len(tra) - 1][1] - tra[1][1]
            else:
                slope = tra[len(tra) - 1][2] - tra[0][2]
                dif_x = tra[len(tra) - 1][1] - tra[0][1]

            if (slope < 0):         #North
                if (dif_x > 0):     #East
                    direction.append("NE")
                elif (dif_x < 0):   #West
                    direction.append("NW")
                else:               #Only vertical movement
                    direction.append("N")
            elif (slope > 0):       #South
                if (dif_x > 0):     #East
                    direction.append("SE")
                elif (dif_x < 0):   #West
                    direction.append("SW")
                else:               #Only vertical movement
                    direction.append("S")
            else:                   #Only horizontal movement
                if (dif_x > 0):     #East
                    direction.append("E")
                elif (dif_x < 0):   #West
                    direction.append("W")
                else:               #First and last point are the same
                    direction.append("X")

            zones_w_dir[i][-1].append(direction)

    #Write zones with the speeds in a file
    file = file_preffix + "_complete_zones.txt"
    with open(file, 'w+') as file:
        for zone in zones_w_dir:
            file.write(str(zone) + "\n\n")

    return zones_w_dir

#-----------------------------------------------------------------------------#
#Gaussian distribution of speeds
#To calculate mean and standard deviation of the speeds, we ignore trajectories
#with speed = 0 that last less than 20 frames
#-----------------------------------------------------------------------------#
def gaussian_dist (zones, file_preffix):

    #Copy the information already known about the zones
    zones_dist = deepcopy(zones)

    means = []
    devs = []

    #Calculate mean and standard deviation of each zone
    for zone in zones:
        speeds = []
        for tra_speed in zone[3]:
            speeds.append(tra_speed[0])

        means.append(numpy.mean(speeds))
        devs.append(numpy.std(speeds))

    #Store mean and standard deviation of each zone
    for i in range (0, len(zones_dist)):
        zones_dist[i].append(means[i])
        zones_dist[i].append(devs[i])

    #Write zones with mean and standard deviation in a file
    file = file_preffix + "_zones_distribution.txt"
    with open(file, 'w+') as file:
        for zone in zones_dist:
            file.write(str(zone) + "\n\n")

    return zones_dist

#-----------------------------------------------------------------------------#
#Detect main direction of each zone
#-----------------------------------------------------------------------------#
def main_direction (zones, file_preffix):

    #Copy the information already known about the zones
    zones_main_direction = deepcopy(zones)

    i = 0

    #For each zone, store the frequency of each direction
    for zone in zones:

        observed = {"N": 0, "NE": 0, "NW": 0, "S": 0, "SE": 0, "SW": 0, "E": 0, "W": 0, "X": 0}

        for direction in zone[4]:
            observed[direction[0]] += 1

        pearson_chi_sq = 0

        max = ["", 0]

        snd_max = ["", 0]

        main_directions = ["None", "None"]

        for direction, frequency in observed.items():

            #Calculate Pearson's chi-squared test
            pearson_chi_sq += pow((frequency - 1/NUM_DIRECTIONS), 2) / (1/NUM_DIRECTIONS)

            #Save the two most frequent directions
            if (frequency > max[1]):
                snd_max[0] = max[0]
                snd_max[1] = max[1]
                max[0] = direction
                max[1] = frequency
            else:
                if (frequency > snd_max[1]):
                    snd_max[0] = direction
                    snd_max[1] = frequency

        #If Pearson's chi-squared test is 0, the observed distribution is uniform
        if (pearson_chi_sq != 0):

            #If there is only one direction, that is the main direction
            if (snd_max[0] == ""):
                main_directions = [max[0], max[0]]
            #If most frequent direction is X, the second most frequent does not matter
            elif (max[0] == "X"):
                main_directions = ["X", "X"]
            #If the second most frequent is X, it does not matter
            elif (snd_max[0] == "X"):
                main_directions = [max[0], max[0]]
            else:
                #If the two most frequent directions are opposite
                if (snd_max[0] in OPPOSITE_DIRECTIONS[max[0]]):
                    #If they have similar probability, store two as main
                    if ((max[1] - snd_max[1]) <= (max[1]//2)):
                        main_directions = [max[0], snd_max[0]]
                    #If they haven't got similar probabilities, store only max
                    else:
                        main_directions = [max[0], max[0]]
                #If the two most frequent directions are not opposite, the main
                #direction is the composition of them
                else:
                    main = ""
                    for compound, directions in COMPOUND_DIRECTIONS.items():
                        if (max[0] in directions and snd_max[0] in directions):
                            main = compound
                    main_directions = [main, main]

        zones_main_direction[i].append(main_directions)
        i += 1

    #Write main direction of each zone in a file
    file = file_preffix + "_zones_main_direction.txt"
    with open(file, 'w+') as file:
        for zone in zones_main_direction:
            for z in zone:
                file.write("*" + str(z) + "\n")
            file.write("\n")

    return zones_main_direction

#-----------------------------------------------------------------------------#
#Add default zone
#-----------------------------------------------------------------------------#
def default_zone (zones):

    zones_default = deepcopy(zones)

    covered_cells = []

    cells_zones = {z: [] for z in range(len(zones))}

    #Store which cells belong to any/each zone
    for zone in zones:
        covered_cells.extend(zone[1])
        cells_zones[zone[0]] = zone[1]

    uncovered_cells = []

    #For each cell in the grid
    for i in range(0, ((NUM_CELLS_HOR * NUM_CELLS_VER))):
        #If the cell does not belong to any zone
        if (not i in covered_cells):
            neighbors_zones = [-1]*8
            neighbors = [i-NUM_CELLS_HOR-1, i-NUM_CELLS_HOR, i-NUM_CELLS_HOR+1,
                        i-1, i+1, i+NUM_CELLS_HOR-1, i+NUM_CELLS_HOR,
                        i+NUM_CELLS_HOR+1]

            #Store to which zone belongs each neighbor
            for n in range(len(neighbors)):
                if (neighbors[n] in covered_cells):
                    for id, cells in cells_zones.items():
                        if (neighbors[n] in cells):
                            neighbors_zones[n] = id

            #Check if, at least, 5 neighbors are from the same zone and, in that
            #case, append current cell to that specific zone. Else, leave it in
            #the default zone
            neighbors_zones_counter = Counter(neighbors_zones)

            if (neighbors_zones_counter.most_common(1)[0][1] >= 5 and
                neighbors_zones_counter.most_common(1)[0][0] != -1):

                most_repeated_zone = neighbors_zones_counter.most_common(1)[0][0]
                zones_default[most_repeated_zone][1].append(i)
                covered_cells.append(i)
                cells_zones[most_repeated_zone].append(i)
            else:
                uncovered_cells.append(i)

    zones_default.append([-1, uncovered_cells, []])

    #Write zones with abnormal directions in a file
    file = train_result_folder + "train_zones_default.txt"
    with open(file, 'w+') as file:
        for zone in zones_default:
            for z in zone:
                file.write("*" + str(z) + "\n")
            file.write("\n")

    return zones_default

#-----------------------------------------------------------------------------#
#Add speed and direction to default zone
#-----------------------------------------------------------------------------#
def default_zone_sp_dir (zones_train, zones_default):

    zones_train_default = deepcopy(zones_train)

    for zone_id in range(len(zones_train_default)):
        zones_train_default[zone_id][1] = zones_default[zone_id][1]

    if (zones_default[-1][0] == -1):
        zones_train_default.append([-1, zones_default[-1][1], [], [], []])

        speed = 0
        st_dev = 0

        for zone in zones_train:
            speed += zone[5]
            st_dev += zone[6]

        zones_train_default[-1].append(speed/len(zones_train))
        zones_train_default[-1].append(st_dev/len(zones_train))
        zones_train_default[-1].append(['Def', 'Def'])

    #Write zones with abnormal directions in a file
    file = train_result_folder + "train_main_direction_default.txt"
    with open(file, 'w+') as file:
        for zone in zones_train_default:
            for z in zone:
                file.write("*" + str(z) + "\n")
            file.write("\n")

    return zones_train_default
