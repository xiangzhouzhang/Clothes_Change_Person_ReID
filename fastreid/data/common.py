# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):

        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


class CCDatasets(Dataset):
    """For Clothes Changing datasets"""
    def __init__(self, img_items, transform=None, relabel=True):
            self.img_items = img_items
            self.transform = transform
            self.relabel = relabel
            
            pid_set = set()
            cam_set = set()
            clo_set = set()
            for i in img_items:
                pid_set.add(i[1])
                cam_set.add(i[2])
                clo_set.add(i[3])
            
            self.pids = sorted(list(pid_set))
            self.cams = sorted(list(cam_set))
            self.clos = sorted(list(clo_set))
            if relabel:
                self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
                self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
                self.clo_dict = dict([(p, i) for i, p in enumerate(self.clos)])


    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        
        img_path, pid, camid, cloid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None: 
            img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
            cloid = self.clo_dict[cloid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "clothids": cloid,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)

    @property
    def num_clothes(self):
        return len(self.clos)

