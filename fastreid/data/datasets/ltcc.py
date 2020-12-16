"""
@author:  xiangzhou
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from tabulate import tabulate
from termcolor import colored
import copy
import logging
import os

logger = logging.getLogger(__name__)

@DATASET_REGISTRY.register()
class LTCC(ImageDataset):
    """LTCC.

    Dataset statistics:
        - identities: 77 + 75 (+1 for background).
        - images: 9,576  (train) + 493  (query).
    """

    # dataset_dir = 'LTCC_ReID/'
    dataset_url = None
    dataset_name = "LTCC"

    def __init__(self, root='datasets', **kwargs):

        # import pdb; pdb.set_trace()
        self.root = root
      
        data_dir = osp.join(self.root, 'LTCC_ReID')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated.')

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'test')
        self.info = osp.join(self.data_dir, 'info')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.info,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)
        
        self.num_train_clos = self.get_num_clos(train)

        super(LTCC, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))

        pattern = re.compile(r'([-\d]+)_([-\d]+)_c([-\d]+)')

        data = []
        for img_path in img_paths:
            pid, cloid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 151  # pid == 0 means background  (but we do not have background class)
            assert 1 <= camid <= 12
            assert cloid >= 1
            camid -= 1  # index starts from 0
            cloid -= 1  # index starts from 0
            # if is_train:
            #     pid = self.dataset_name + "_" + str(pid)
            #     camid = self.dataset_name + "_" + str(camid)
            #     cloid = str(pid) + "_" + str(cloid)
            pid = self.dataset_name + "_" + str(pid)
            camid = self.dataset_name + "_" + str(camid)
            cloid = str(pid) + "_" + str(cloid)
           
            data.append((img_path, pid, camid, cloid))
        return data
    
    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        clos = set()
        for _, pid, camid, cloid in data:
            pids.add(pid)
            cams.add(camid)
            clos.add(cloid)
            
        return len(pids), len(cams), len(clos)
    
    def get_num_clos(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[2]


    # def combine_all(self):
    #     """Combines train, query and gallery in a dataset for training."""
    #     combined = copy.deepcopy(self.train)

    #     def _combine_data(data):
    #         for img_path, pid, camid, cloid in data:
    #             pid = self.dataset_name + "_" + str(pid)
    #             camid = self.dataset_name + "_" + str(camid)
    #             cloid = self.dataset_name + "_" + str(cloid)
    #             combined.append((img_path, pid, camid, cloid))

    #     _combine_data(self.query)
    #     _combine_data(self.gallery)

    #     self.train = combined
    #     self.num_train_pids = self.get_num_pids(self.train)

    
    def show_train(self):
        num_train_pids, num_train_cams, num_train_clos = self.parse_data(self.train)

        headers = ['subset', '# ids', '# images', '# cameras']
        csv_results = [['train', num_train_pids, len(self.train), num_train_cams]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))

    def show_test(self):
        num_query_pids, num_query_cams, num_query_clos = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams, num_gallery_clos = self.parse_data(self.gallery)

        headers = ['subset', '# ids', '# images', '# cameras']
        csv_results = [
            ['query', num_query_pids, len(self.query), num_query_cams],
            ['gallery', num_gallery_pids, len(self.gallery), num_gallery_cams],
        ]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))