# -*- coding: utf-8 -*-
"""
Laura Hernández Muñoz

Draw detections and ground truth bounding boxes in the same video
"""

from utils import *

gt_root = "./gt/"
det_folder = "./test_track/"

for test in os.listdir(gt_root):
    if not os.path.isfile(os.path.join(gt_root, test)):
        test_gt_folder = gt_root + test + "/"
        for gt_folder in os.listdir(test_gt_folder):
            if not os.path.isfile(os.path.join(test_gt_folder, gt_folder)):
                gt = test_gt_folder + gt_folder + "/"
                for file in os.listdir(gt):
                    if os.path.isfile(os.path.join(gt, file)) and file.endswith("gt.txt"):

                        gt_path = gt + file
                        gt_file = open (gt_path, "r")

                        #Read ground truth information
                        gt_trs = gt_file.readlines()

                        #Empty list for the trajectories (gt)
                        gt_trajectories = []

                        #For each line from the input file
                        for tr in gt_trs:

                            #Ignore blank lines
                            if (tr != "\n"):

                                split_tr = tr.split(",")

                                frame_id = split_tr[0]
                                tr_id = split_tr[1]
                                x_left = split_tr[2]
                                y_top = split_tr[3]
                                width = split_tr[4]
                                height = split_tr[5]

                                gt_trajectories.append([frame_id, tr_id, x_left, y_top, width, height])

                        gt_file.close()

                        det_path = det_folder + test + ".txt"

                        det_file = open (det_path, "r")

                        #Read detections information
                        det_trs = det_file.readlines()

                        #Empty list for the trajectories (detections)
                        det_trajectories = []

                        #For each line from the input file
                        for tr in det_trs:

                            #Ignore blank lines
                            if (tr != "\n"):

                                split_tr = tr.split(",")

                                frame_id = split_tr[0]
                                tr_id = split_tr[1]
                                x_left = split_tr[2]
                                y_top = split_tr[3]
                                width = split_tr[4]
                                height = split_tr[5]

                                det_trajectories.append([frame_id, tr_id, x_left, y_top, width, height])

                        det_file.close()

                        #Draw the bounding boxes of the abnormal trajectories
                        #(To work on CCTV change "peds1_test" to "cctv_test") NOT AVAILABLE
                        dataset = Datasets("peds1_test")
                        out_dir = "./pruebagt/"

                        try:
                            os.stat(out_dir)
                        except:
                            os.mkdir(out_dir)

                        for seq in dataset:
                            if (str(seq) == test):

                                output_dir = out_dir + test + "/"

                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)

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

                                    #Check which trajectories are in the current frame and draw them (gt)
                                    for tr in gt_trajectories:
                                        if ((i + 1) == int(tr[0])):
                                            ax.add_patch(
                                                plt.Rectangle(
                                                    (float(tr[2]), float(tr[3])),
                                                    float(tr[4]),
                                                    float(tr[5]),
                                                    fill=False,
                                                    linewidth=1.0, edgecolor = 'green'))

                                            ax.annotate(int(tr[1]), (float(tr[2]) + float(tr[4]) / 2.0,
                                                                     float(tr[3]) + float(tr[5]) / 2.0),
                                                        color='green', weight='bold',
                                                        fontsize=6, ha='center', va='center')

                                    #Check which trajectories are in the current frame and draw them (det)
                                    for tr in det_trajectories:
                                        if ((i + 1) == int(tr[0])):
                                            ax.add_patch(
                                                plt.Rectangle(
                                                    (float(tr[2]), float(tr[3])),
                                                    float(tr[4]),
                                                    float(tr[5]),
                                                    fill=False,
                                                    linewidth=1.0, edgecolor = 'red'))

                                            ax.annotate(int(tr[1]), (float(tr[2]) + float(tr[4]) / 2.0,
                                                                     float(tr[3]) + float(tr[5]) / 2.0),
                                                        color='red', weight='bold',
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

                                out_video = test + "_det_gt.avi"
                                out = cv2.VideoWriter(osp.join(output_dir, out_video),
                                                      cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

                                for i in range(len(img_array)):
                                    out.write(img_array[i])
                                out.release()
