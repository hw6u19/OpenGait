import math
import random
import numpy as np
from ..utils import get_msg_mgr


class CollateFn(object):
    def __init__(self, label_set, sampler_config):
        self.label_set = label_set
        sample_type = sampler_config['sample_type']
        sample_type = sample_type.split('_')
        self.sampler = sample_type[0]
        self.ordered = sample_type[1]
        if self.sampler not in ['fixed', 'unfixed', 'all']:
            raise ValueError(f'sampler type should be fixed, unfixed or all but got {self.sampler}')
        if self.ordered not in ['ordered', 'unordered']:
            raise ValueError(f'sampler type should be ordered or unordered, but got {self.ordered}')
        # transfer value of self.ordered to boolean value True or False
        self.ordered = sample_type[1] == 'ordered'

        # fixed cases
        if self.sampler == 'fixed':
            self.frames_num_fixed = sampler_config['frames_num_fixed']

        # unfixed cases
        if self.sampler == 'unfixed':
            self.frames_num_max = sampler_config['frames_num_max']
            self.frames_num_min = sampler_config['frames_num_min']

        if self.sampler != 'all' and self.ordered:
            self.frames_skip_num = sampler_config['frames_skip_num']

        self.frames_all_limit = -1
        if self.sampler == 'all' and 'frames_all_limit' in sampler_config:
            self.frames_all_limit = sampler_config['frames_all_limit']
        
        self.count = 0
        self.feature_num = 0
        self.batch_size = 0

        self.seqs_batch = None
        self.labels_batch = None
        self.types_batch = None
        self.views_batch = None
        self.frames_batch = None
        
    def __call__(self, batch: list):
        self.batch_size = len(batch)
        # currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        self.feature_num = len(batch[0][0])
        # sequence, label, type, view
        self.seqs_batch, self.labels_batch, self.types_batch, self.views_batch = [], [], [], []

        for bt in batch:
            # bt is the returned value of dataset __getitem__，[data_list, seq_info]
            # bt[0] is data_list [data(numpy.ndarray)]
            # bt[1] is seq_info [label, type, view, [related .pkl file paths]]
            self.seqs_batch.append(bt[0])
            # list.index(value) is used to find the index of the first occurrence of a specified value in a list
            self.labels_batch.append(self.label_set.index(bt[1][0]))
            self.types_batch.append(bt[1][1]) # [batch_size, 1]
            self.views_batch.append(bt[1][2])

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num
        self.frames_batch = [self._sample_frames(seqs) for seqs in self.seqs_batch]  # [batch, feature_num, sequence_length]
        batch = [self.frames_batch, self.labels_batch, self.types_batch, self.views_batch, None]

        batch = self._batch_reshape(batch)

        return batch

    def _frames_num_check(self):
        """
        Returns:
            number of frames based on the type of sampler, fixed or unfixed
        """
        if self.sampler == 'fixed':
            return self.frames_num_fixed
        else:
            return random.choice(
                list(range(self.frames_num_min, self.frames_num_max + 1)))

    def _sample_frames(self, seqs: list):
        """
        Sample frames from data_list based on the number of frames get from config file. The requested sequence
        length may be different with the total number of frames in the data_list.

        Ordered: the sequences should follow the original order

        Unordered: the sequence should not follow the original order, and the samples are randomly chosen and can
        be duplicated

        Args:
            seqs: list of data, which is List[numpy.ndarray]

        Returns:
            List[List[numpy.ndarray(HxW)]] sampled_frames, in the similar shape to data_list, which is
            List[numpy.ndarray(LxHxW)].
        """
        sampled_frames = [[] for i in range(self.feature_num)]
        seq_len = len(seqs[0])
        indices = list(range(seq_len))
        frames_num = self._frames_num_check()

        if self.ordered:
            # ordered
            fs_n = frames_num + self.frames_skip_num
            # make sure that seq_len >= fs_n
            if seq_len < fs_n:
                it = math.ceil(fs_n / seq_len)
                seq_len = seq_len * it
                indices = indices * it

            start_idx = random.choice(list(range(seq_len - fs_n + 1)))
            end_idx = start_idx + fs_n
            idx_lst = list(range(start_idx, end_idx + 1))
            idx_lst = sorted(np.random.choice(
                idx_lst, frames_num, replace=False))
            indices = [indices[i] for i in idx_lst]
        else:
            # unordered
            replace = seq_len < frames_num

            if seq_len == 0:
                get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                        % (str(self.labels_batch[self.count]), str(self.types_batch[self.count]), str(self.views_batch[self.count])))

            self.count += 1
            indices = np.random.choice(
                indices, frames_num, replace=replace)

        for i in range(self.feature_num):
            # sampled_frames[i] is List[numpy.ndarray]
            for j in indices[:self.frames_all_limit] if -1 < self.frames_all_limit < len(indices) else indices:
                sampled_frames[i].append(seqs[i][j])
        return sampled_frames

    def _batch_reshape(self, batch:list):
        """
        Reshape the frames_batch based on the number of frames is whether fixed or unfixed.
        """
        if self.sampler == "fixed":
            # transfer frames_batch[i][j] from a list to numpy.ndarray
            self.frames_batch = [[np.asarray(self.frames_batch[i][j]) for i in range(self.batch_size)]
                          for j in range(self.feature_num)]  # [f, b, 1], (sequence_length, H, W)
        else:
            sequence_length_batch = [[len(self.frames_batch[i][0]) for i in range(self.batch_size)]]  # [1, b]

            def my_cat(k):
                return np.concatenate([self.frames_batch[i][k] for i in range(self.batch_size)], 0)

            self.frames_batch = [[my_cat(k)] for k in range(self.feature_num)]  # [f, 1], (g, H, W), g is the summarized sequence length in the batch

            batch[-1] = np.asarray(sequence_length_batch)

        batch[0] = self.frames_batch
        return batch