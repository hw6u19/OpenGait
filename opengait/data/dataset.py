import os
import pickle
import os.path as osp
from typing import Tuple

import torch.utils.data as torchdata
import json
from ..utils import get_msg_mgr


class DataSet(torchdata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.__dataset_parser(data_cfg, training)
        # cache decides whether you want to load all data into cache for quicker loading, at the cost of more memory used
        self.cache = data_cfg['cache']
        # seq_info is in the format of [label, type, view, [related .pkl file paths]]
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]
        # use set() to delete duplicated labels, types and views
        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        # [None, None, ..., None]
        # {'001':[], '002':[], '003':[], ...}
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}

        for i, seq_info in enumerate(self.seqs_info):
            # seq_info[0] is label
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths: list[str]) -> list:
        """
        Args:
            paths (list[str]): A list of absolute paths of pickle files.
        Returns:
            A list of related data saved in the pickle files.
        """
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    data = pickle.load(f)
                f.close()
                data_list.append(data)
            else:
                raise ValueError('- Loader - only support .pkl !!!')

        # validation of  the length of loaded data
        # Each input data should have the same length
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx: int) -> Tuple:
        """
        Args:
            idx (int): Index
        Returns:
            A Tuple consisting of data_list and seq_info for the given index.
        """
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])

        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        # root path of dataset
        dataset_root = data_config['dataset_root']

        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        # dataset train-test subject number partition
        # ideal partition, which includes all subject name in the dataset
        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]

        # check if some subjects' data is missed in the dataset
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]

        # message manager
        msg_mgr = get_msg_mgr()

        # log the information about the pid_list
        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            # label is subject name. such as '001'
            for label in label_set:
                # type is the covariate, such as 'normal', 'backpack', etc
                for type in sorted(os.listdir(osp.join(dataset_root, label))):
                    # view is the camera view
                    for view in sorted(os.listdir(osp.join(dataset_root, label, type))):
                        seq_info = [label, type, view]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        # find the absolute paths of .pkl files in the dataset_root/label/type/view/
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir) for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (label, type, view))
            # seqs_info_list is the list of [label, type, view, [related pkl paths]]
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(train_set) if training else get_seqs_info_list(test_set)
