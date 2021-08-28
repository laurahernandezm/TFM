# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Functions related to anomaly detection
"""

from utils import *

#-----------------------------------------------------------------------------#
# Detect abnormal speeds
#-----------------------------------------------------------------------------#
def abnormal_speeds (zones, file_preffix, test_trajectories):

    #Copy the information already known about the zones
    zones_abnormal_speeds = deepcopy(zones)

    #Define confidence intervals for each zone and check each speed with them
    for i in range(0, len(zones)):
        abnormal_speeds = []
        inf_interval = zones[i][5] - (ALPHA*LOW_FACTOR) * zones[i][6]
        sup_interval = zones[i][5] + (ALPHA*HIGH_FACTOR) * zones[i][6]

        for test_zone in test_trajectories:
            if (test_zone[0] == zones[i][0] and test_zone[1][0] == zones[i][1][0]):

                for tra_speed in test_zone[2]:
                    if (tra_speed[1] > sup_interval or tra_speed[1] < inf_interval):
                        abnormal_speeds.append([tra_speed[0], tra_speed[1]])
                zones_abnormal_speeds[i].append(abnormal_speeds)

    #Delete zones without information
    delete = []
    for j in range(0, len(zones_abnormal_speeds)):
        if (len(zones_abnormal_speeds[j]) < 9):
            delete.append(j)

    delete.sort(reverse = True)

    for d in delete:
        del zones_abnormal_speeds[d]

    #Write zones with abnormal speeds in a file
    file = file_preffix + "_zones_abnormal_speeds.txt"
    with open(file, 'w+') as file:
        for zone in zones_abnormal_speeds:
            for z in zone:
                file.write("*" + str(z) + "\n")
            file.write("\n")

    return zones_abnormal_speeds

#-----------------------------------------------------------------------------#
# Filter abnormal trajectories according to their speed
#-----------------------------------------------------------------------------#
def filter_trajectories_speed (file, zones, file_preffix):

    abnormal_trajectories = set()

    for zone in zones:
        for abnormal_tra in zone[2]:

            abnormal_trajectories.add(int(abnormal_tra[0]))

    abnormal_trajectories = sorted(abnormal_trajectories)

    #Input file with MOT format (frame, id, bb_left, bb_top, bb_width, bb_height)
    file_input = open (file, "r")
    #Output file with MOT format (frame, id, bb_left, bb_top, bb_width, bb_height)
    file_o = file_preffix + "_filtered_trajectories_speed.txt"
    file_output = open (file_o, "w")

    #Read all lines from input file
    trs = file_input.readlines()

    #Empty list for the trajectories
    trajectories = []

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

            #If the trajectory is abnormal, save it in the output file
            if (int(tr_id) in abnormal_trajectories):
                trajectories.append([frame_id, tr_id, x_left, y_top, width, height])
                file_output.write(frame_id + "," + tr_id + "," + x_left + "," +
                                  y_top + "," + width + "," + height + "," +
                                  "-1,-1,-1,-1\n")

    file_input.close()
    file_output.close()

    #Draw only the bounding boxes of the abnormal trajectories
    dataset = Datasets("peds1_test")
    out_dir = "./abnormal_speeds/"

    try:
        os.stat(out_dir)
    except:
        os.mkdir(out_dir)

    last_bar = -1
    for x, v in enumerate(file_preffix):
        if (v == '/'):
            last_bar = x
    last_bar += 1

    match = file_preffix[last_bar:]

    for seq in dataset:
        if (str(seq) == match):

            output_dir = out_dir + match + "/"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            #Color loop
            cyl = cy('ec', my_colors.CSS4_COLORS)
            loop_cy_iter = cyl()
            styles = defaultdict(lambda: next(loop_cy_iter))

            #For each frame
            for i, v in enumerate(seq):
                im_path = v['img_path']
                im_name = osp.basename(im_path)
                im_output = osp.join(output_dir, im_name)
                im = cv2.imread(im_path)
                im = im[:, :, (2, 1, 0)]

                sizes = numpy.shape(im)
                height = float(sizes[0])
                width = float(sizes[1])

                fig = plt.figure()
                fig.set_size_inches(width / 100, height / 100)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(im)

                #Check which trajectories are in the current frame and draw them
                for tr in trajectories:
                    if ((i + 1) == int(tr[0])):
                        ax.add_patch(
                            plt.Rectangle(
                                (float(tr[2]), float(tr[3])),
                                float(tr[4]),
                                float(tr[5]),
                                fill=False,
                                linewidth=1.0, **styles[tr[1]]))

                        ax.annotate(int(tr[1]), (float(tr[2]) + float(tr[4]) / 2.0,
                                                 float(tr[3]) + float(tr[5]) / 2.0),
                                    color=styles[tr[1]]['ec'], weight='bold',
                                    fontsize=6, ha='center', va='center')

                plt.axis('off')
                plt.draw()
                plt.savefig(im_output, dpi=100)
                plt.close()

            #Transform the sequence of frames into a video
            img_array = []
            dir = osp.join(output_dir, "*.jpg")
            files = glob.glob(dir)
            sorted_files = natsorted (files)

            for filename in sorted_files:
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)

            out_video = str(file_preffix[last_bar:]) + "_result_video_abnormal_speed.avi"
            out = cv2.VideoWriter(osp.join(output_dir, out_video),
                                  cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()

#-----------------------------------------------------------------------------#
#Detect abnormal directions
#-----------------------------------------------------------------------------#
def abnormal_directions (zones, file_preffix, test_trajectories):

    #Copy the information already known about the zones
    zones_abnormal_directions = deepcopy(zones)

    #For each zone, check which trajectories are opposite to the main direction
    for i in range(0, len(zones)):

        if (zones[i][7][0] != "None" and zones[i][7][1] != "None"):
            abnormal_directions = []
            for tra_dir in test_trajectories[i][3]:
                if (tra_dir[1] in OPPOSITE_DIRECTIONS[zones[i][7][0]] and
                    tra_dir[1] in OPPOSITE_DIRECTIONS[zones[i][7][1]]):
                    abnormal_directions.append([tra_dir[0], tra_dir[1]])
            zones_abnormal_directions[i].append(abnormal_directions)
        else:
            zones_abnormal_directions[i].append("No direction analysis performed")

    #Write zones with abnormal directions in a file
    file = file_preffix + "_zones_abnormal_directions.txt"
    with open(file, 'w+') as file:
        file.write("0: zone id\n1: cells belonging to the zone\n")
        file.write("2: snapped trajectories in the zone\n3: speeds\n4: directions\n")
        file.write("5: speeds' mean\n6: speeds' deviation\n7: main direction\n")
        file.write("8: abnormal speeds\n9: abnormal directions\n\n")
        for zone in zones_abnormal_directions:
            for z in zone:
                file.write("*" + str(z) + "\n")
            file.write("\n")

    return zones_abnormal_directions

#-----------------------------------------------------------------------------#
# Filter abnormal trajectories according to their direction
#-----------------------------------------------------------------------------#
def filter_trajectories_direction (file, zones, file_preffix):

    abnormal_trajectories = set()

    for zone in zones:
        if (zone[1][0] != "None"):
            for abnormal_tra in zone[3]:
                abnormal_trajectories.add(int(abnormal_tra[0]))

    abnormal_trajectories = sorted(abnormal_trajectories)

    #Input file with MOT format (frame, id, bb_left, bb_top, bb_width, bb_height)
    file_input = open (file, "r")
    #Output file with MOT format (frame, id, bb_left, bb_top, bb_width, bb_height)
    file_o = file_preffix + "_filtered_trajectories_direction.txt"
    file_output = open (file_o, "w")

    #Read all lines from input file
    trs = file_input.readlines()

    #Empty list for the trajectories
    trajectories = []

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

            #If the trajectory is abnormal, save it in the output file
            if (int(tr_id) in abnormal_trajectories):
                trajectories.append([frame_id, tr_id, x_left, y_top, width, height])
                file_output.write(frame_id + "," + tr_id + "," + x_left + "," +
                                  y_top + "," + width + "," + height + "," +
                                  "-1,-1,-1,-1\n")

    file_input.close()
    file_output.close()

    #Draw only the bounding boxes of the abnormal trajectories
    dataset = Datasets("peds1_test")
    out_dir = "./abnormal_directions/"

    try:
        os.stat(out_dir)
    except:
        os.mkdir(out_dir)

    last_bar = -1
    for x, v in enumerate(file_preffix):
        if (v == '/'):
            last_bar = x
    last_bar += 1

    match = file_preffix[last_bar:]

    for seq in dataset:

        if (str(seq) == match):

            output_dir = out_dir + match + "/"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            #Color loop
            cyl = cy('ec', my_colors.CSS4_COLORS)
            loop_cy_iter = cyl()
            styles = defaultdict(lambda: next(loop_cy_iter))

            #For each frame
            for i, v in enumerate(seq):
                im_path = v['img_path']
                im_name = osp.basename(im_path)
                im_output = osp.join(output_dir, im_name)
                im = cv2.imread(im_path)
                im = im[:, :, (2, 1, 0)]

                sizes = numpy.shape(im)
                height = float(sizes[0])
                width = float(sizes[1])

                fig = plt.figure()
                fig.set_size_inches(width / 100, height / 100)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(im)

                #Check which trajectories are in the current frame and draw them
                for tr in trajectories:
                    if ((i + 1) == int(tr[0])):
                        ax.add_patch(
                            plt.Rectangle(
                                (float(tr[2]), float(tr[3])),
                                float(tr[4]),
                                float(tr[5]),
                                fill=False,
                                linewidth=1.0, **styles[tr[1]]))

                        ax.annotate(int(tr[1]), (float(tr[2]) + float(tr[4]) / 2.0,
                                                 float(tr[3]) + float(tr[5]) / 2.0),
                                    color=styles[tr[1]]['ec'], weight='bold',
                                    fontsize=6, ha='center', va='center')

                plt.axis('off')
                plt.draw()
                plt.savefig(im_output, dpi=100)
                plt.close()

            #Transform the sequence of frames into a video
            img_array = []
            dir = osp.join(output_dir, "*.jpg")
            files = glob.glob(dir)
            sorted_files = natsorted (files)

            for filename in sorted_files:
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)

            out_video = str(file_preffix[last_bar:]) + "_result_video_abnormal_direction.avi"
            out = cv2.VideoWriter(osp.join(output_dir, out_video),
                                  cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()

#-----------------------------------------------------------------------------#
# Filter abnormal trajectories according to their speed and direction
#-----------------------------------------------------------------------------#
def filter_trajectories_both (file, zones, file_preffix, x_min, y_min, x_max, y_max):

    abnormal_trajectories = set()

    for zone in zones:
        for abnormal_tra in zone[2]:
            abnormal_trajectories.add(int(abnormal_tra[0]))

    for zone in zones:
        if (zone[1][0] != "None"):
            for abnormal_tra in zone[3]:
                abnormal_trajectories.add(int(abnormal_tra[0]))

    abnormal_trajectories = sorted(abnormal_trajectories)

    #Input file with MOT format (frame, id, bb_left, bb_top, bb_width, bb_height)
    file_input = open (file, "r")
    #Output file with MOT format (frame, id, bb_left, bb_top, bb_width, bb_height)
    file_o = file_preffix + "_filtered_trajectories_both.txt"
    file_output = open (file_o, "w")

    #Read all lines from input file
    trs = file_input.readlines()

    #Empty list for the trajectories
    trajectories = []

    #For each line from the input file
    for tr in trs:

        #Ignore blank lines
        if (tr != "\n"):

            split_tr = tr.split(",")

            frame_id = split_tr[0]
            tr_id = split_tr[1]
            x_left = float(split_tr[2])
            y_top = float(split_tr[3])
            width = float(split_tr[4])
            height = float(split_tr[5])

            #If the trajectory is abnormal, save it in the output file
            if (int(tr_id) in abnormal_trajectories):

                #If the bbox goes out of the frame on the left
                if (x_left < x_min):
                    #Set x value to the left limit of the frame and re-compute
                    #width
                    x_left = x_min
                    width = width - x_left

                #If the bbox goes out of the frame on the right
                if (x_left + width > x_max):
                    #Adjust width so the bbox stays inside the frame
                    width = x_max - x_left

                #If the bbox goes out of the frame on the top
                if (y_top < y_min):
                    #Set y value to the upper limit of the frame and re-compute
                    #height
                    y_top = y_min
                    height = height - y_top

                #If the bbox goes out of the frame on the bottom
                if (y_top + height > y_max):
                    #Adjust height so the bbox stays inside the frame
                    height = y_max - y_top

                trajectories.append([frame_id, tr_id, x_left, y_top, width, height])
                file_output.write(frame_id + "," + tr_id + "," + str(x_left) + "," +
                                  str(y_top) + "," + str(width) + "," + str(height) + "," +
                                  "-1,-1,-1,-1\n")

    file_input.close()
    file_output.close()

    #Draw only the bounding boxes of the abnormal trajectories
    '''dataset = Datasets("peds1_test")
    out_dir = "./abnormal_both/"

    try:
        os.stat(out_dir)
    except:
        os.mkdir(out_dir)

    last_bar = -1
    for x, v in enumerate(file_preffix):
        if (v == '/'):
            last_bar = x
    last_bar += 1

    match = file_preffix[last_bar:]

    for seq in dataset:

        if (str(seq) == match):

            output_dir = out_dir + match + "/"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            #Color loop
            cyl = cy('ec', my_colors.CSS4_COLORS)
            loop_cy_iter = cyl()
            styles = defaultdict(lambda: next(loop_cy_iter))

            #For each frame
            for i, v in enumerate(seq):
                im_path = v['img_path']
                im_name = osp.basename(im_path)
                im_output = osp.join(output_dir, im_name)
                im = cv2.imread(im_path)
                im = im[:, :, (2, 1, 0)]

                sizes = numpy.shape(im)
                height = float(sizes[0])
                width = float(sizes[1])

                fig = plt.figure()
                fig.set_size_inches(width / 100, height / 100)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(im)

                #Check which trajectories are in the current frame and draw them
                for tr in trajectories:
                    if ((i + 1) == int(tr[0])):
                        ax.add_patch(
                            plt.Rectangle(
                                (float(tr[2]), float(tr[3])),
                                float(tr[4]),
                                float(tr[5]),
                                fill=False,
                                linewidth=1.0, **styles[tr[1]]))

                        ax.annotate(int(tr[1]), (float(tr[2]) + float(tr[4]) / 2.0,
                                                 float(tr[3]) + float(tr[5]) / 2.0),
                                    color=styles[tr[1]]['ec'], weight='bold',
                                    fontsize=6, ha='center', va='center')

                plt.axis('off')
                plt.draw()
                plt.savefig(im_output, dpi=100)
                plt.close()

            #Transform the sequence of frames into a video
            img_array = []
            dir = osp.join(output_dir, "*.jpg")
            files = glob.glob(dir)
            sorted_files = natsorted (files)

            for filename in sorted_files:
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)

            out_video = str(file_preffix[last_bar:]) + "_result_video_abnormal_both.avi"
            out = cv2.VideoWriter(osp.join(output_dir, out_video),
                                  cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()'''
