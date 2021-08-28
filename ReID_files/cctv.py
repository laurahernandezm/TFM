import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['CCTV',]


@DATASET_REGISTRY.register()
class CCTV(ImageDataset):

    dataset_dir = "CCTV"
    dataset_name = "cctv"

    def __init__(self, root='datasets', **kwargs):
        self.root = root

        self.data_dir = os.path.join(self.root, self.dataset_dir)

        self.train_dir = os.path.join(self.data_dir, 'train_set')
        self.query_dir = os.path.join(self.data_dir, 'query')
        self.gallery_dir = os.path.join(self.data_dir, 'test_set')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]

        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train = False)
        gallery = self.process_dir(self.gallery_dir, is_train = False)

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, path, is_train = True):
        data = []

        pid_path = path
        pid_list = os.listdir(pid_path)

        for pid_name in pid_list:
            if is_train:
                pid = self.dataset_name + '_' + pid_name[-3:]
            else:
                pid = int(pid_name[-3:])

            cam_list = os.listdir(os.path.join(pid_path, pid_name)) #['cenit', 'esc', 'fac']
            for cam in cam_list:
                cam_num = 1
                if cam == "esc":
                    cam_num = 2
                elif cam == "fac":
                    cam_num = 3

                for cam_folder in os.listdir(os.path.join(pid_path, pid_name, cam)):
                    img_list = glob(os.path.join(pid_path, pid_name, cam, cam_folder, "*.jpg"))

                    for img_path in img_list:
                        img_name = os.path.basename(img_path)
                        if is_train:
                            camid = self.dataset_name + '_' + str(cam_num)
                        else:
                            camid = cam_num

                        data.append([img_path, pid, camid])
        return data
