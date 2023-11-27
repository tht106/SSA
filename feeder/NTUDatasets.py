import numpy as np
import random
import pickle5 as pickle
import cv2
import os
# torch
import torch


def _y_transmat(thetas):
    tms = np.zeros((0, 3, 3))
    thetas = thetas * np.pi / 180
    for theta in thetas:
        tm = np.zeros((3, 3))
        tm[0, 0] = np.cos(theta)
        tm[0, 2] = -np.sin(theta)
        tm[1, 1] = 1
        tm[2, 0] = np.sin(theta)
        tm[2, 2] = np.cos(theta)
        tm = tm[np.newaxis, :, :]
        tms = np.concatenate((tms, tm), axis=0)
    return tms

spectronix_ntu_overlap_classes_for_ntu = {
1: "drink water", 6: "pickup", 7: "throw", 8: "sitting down", 9: "standing up", 10: "clapping",
23: "hand waving", 24: "kicking something", 41: "sneeze/cough",43: "falling",
48: "nausea or vomiting condition", 98: "arm swings"}

def parallel_skeleton(ins_data):
    right_shoulder = ins_data[:, :, 8]  # 9th joint
    left_shoulder = ins_data[:, :, 4]  # 5tf joint
    vec = right_shoulder - left_shoulder
    vec[1, :] = 0
    v = ins_data.shape[2]
    # print(vec.shape)
    l2_norm = np.sqrt(np.sum(np.square(vec), axis=0))
    theta = vec[0, :] / (l2_norm + 0.0001)
    # print(l2_norm)
    thetas = np.arccos(theta) * (180 / np.pi)
    isv = np.sum(vec[2, :])
    if isv >= 0:
        thetas = -thetas
    # print (thetas)
    y_tms = _y_transmat(thetas)
    # print(y_tms)
    new_skel = np.zeros(shape=(0, v, 3))
    # print(new_skel.shape)
    ins_data = ins_data.transpose(1, 2, 0)
    # print(ins_data.shape, new_skel.shape)
    for ind, each_s in enumerate(ins_data):
        # print(each_s.shape)
        r = np.reshape(each_s, newshape=(v, 3))
        r = np.transpose(r)
        r = np.dot(y_tms[ind], r)
        r_t = np.transpose(r)
        r_t = np.reshape(r_t, newshape=(1, -1, 3))
        # print(new_skel.shape, r_t.shape)
        new_skel = np.concatenate((new_skel, r_t), axis=0)
    return new_skel, ins_data


class SimpleLoader(torch.utils.data.Dataset):
    """ Loader for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
    """

    def __init__(self,
                 data_path,
                 label_path,
                 debug=False,
                 mmap=True,
                 data_type='relative',
                 displacement=False,
                 t_length=200,
                 y_rotation=True,
                 sampling='resize'):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.data_type = data_type
        if displacement is not False:
            assert isinstance(displacement, int)
        self.displacement = displacement
        self.max_length = 300
        self.t_length = t_length if t_length < self.max_length else self.max_length
        self.sampling = sampling
        self.y_rotation = y_rotation
        self.mmap = mmap
        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M
        # load data
        position_path = self.data_path.replace('_data.', '_position.')
        motion_path = self.data_path.replace('_data.', '_motion.')
        print(position_path)
        print(motion_path)
        print(self.label_path)

        if mmap:
            self.data = np.load(position_path, mmap_mode='r')
            self.motion = np.load(motion_path, mmap_mode='r') if self.displacement > 0 else None
            self.label = np.load(self.label_path).reshape(-1)
        else:
            self.data = np.load(position_path)
            self.motion = np.load(motion_path) if self.displacement > 0 else None
            self.label = np.load(self.label_path).reshape(-1)
        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        position = np.array(self.data[index])
        motion = np.array(self.motion[index]) if self.displacement > 0 else None
        label = np.array(self.label[index])

        # return motion_data, label
        if motion is not None:
            return position, motion, label
        else:
            return position, label


class NTUMotionProcessor(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
    """

    def __init__(self,
                 data_path,
                 label_path,
                 mmap=True,
                 data_type='normal',
                 t_length=300,
                 y_rotation=False,
                 displacement=False,
                 sampling='force_crop'):
        self.data_path = data_path
        self.label_path = label_path
        self.data_type = data_type
        self.max_length = 300
        self.displacement = displacement
        self.sampling = sampling
        self.y_rotation = y_rotation
        self.t_length = t_length if t_length < self.max_length else self.max_length
        neighbor_1base = [(21, 2), (21, 3), (21, 5), (21, 9), (3, 4),
                          (5, 6), (6, 7), (7, 8), (8, 23), (23, 22),
                          (9, 10), (10, 11), (11, 12), (12, 25), (25, 24),
                          (2, 1), (1, 13), (1, 17), (13, 14), (14, 15),
                          (15, 16), (17, 18), (18, 19), (19, 20)]
        self.neighbor_1base = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M
        # load label
        with open(self.label_path, 'rb') as f:
            print(self.label_path)
            self.sample_name, self.label = pickle.load(f)
            self.label = np.array(self.label)
            self.label = self.label - self.label.min()

        # load data
        print(self.data_path)
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        # index = []
        # label = []
        # for idx, cls in enumerate(self.label):
        #     if cls+1 in list(spectronix_ntu_overlap_classes_for_ntu.keys()):
        #         index.append(idx)
        #         label.append(cls)
        #
        # self.data = self.data[index]
        # self.fps = filter_fps(all_fps, split, class_keys, outer_fp=None, with_join=False)

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        name = self.sample_name[index]
        
        # get motion (displacement)
        if self.displacement > 0:
            motion_data = np.zeros_like(data_numpy)
            motion_data[:, :-self.displacement, :, :] = \
                data_numpy[:, self.displacement:, :, :] - data_numpy[:, :-self.displacement, :, :]
        else:
            motion_data = None

        # get skeleton sequence length, person number
        length = self.get_length(data_numpy)
        num = self.get_person_num(data_numpy)

        # relative coordinate
        if self.data_type == 'relative':
            # center = 21
            C, T, V, M = data_numpy.shape
            center = data_numpy[:, :, 20, :].reshape([C, T, 1, M])
            data_numpy = data_numpy - center
        elif self.data_type == 'normal':
            pass
        else:
            raise TypeError('Invalid data type: %s' % self.data_type)

        # crop length
        if self.sampling == 'force_crop':
            if self.displacement > 0:
                motion_data = motion_data[:, :self.t_length, :, :]
            data_numpy = data_numpy[:, :self.t_length, :, :]
        elif self.sampling == 'resize':
            data_numpy = self.real_resize(data_numpy, length, self.t_length)
            if motion_data is not None:
                motion_data = self.real_resize(motion_data, length, self.t_length)

        else:
            raise TypeError('Invalid sampling type: ' % self.sampling)

        # y rotation
        if self.y_rotation:
            for n in range(2):
                tmp_new, tmp_old = parallel_skeleton(data_numpy[:, :, :, n])
                data_numpy[:, :, :, n] = tmp_new.transpose(2, 0, 1)

        # get actual length
        length = self.t_length if length > self.t_length else length

        # return motion_data, label
        if motion_data is not None:
            return data_numpy, motion_data, label, name
        else:
            return data_numpy, label, name

    @staticmethod
    def get_length(data):
        length = (abs(data[:, :, 0, 0]).sum(axis=0) != 0).sum()
        return length

    @staticmethod
    def get_person_num(data):
        num = (abs(data[:, :, 0, :]).sum(axis=0).sum(axis=0) != 0) * 1.0
        return num

    @staticmethod
    def real_resize(data_numpy, length, crop_size):
        C, T, V, M = data_numpy.shape
        new_data = np.zeros([C, crop_size, V, M])
        for i in range(M):
            tmp = cv2.resize(data_numpy[:, :length, :, i].transpose(
                [1, 2, 0]), (V, crop_size), interpolation=cv2.INTER_LINEAR)
            tmp = tmp.transpose([2, 0, 1])
            new_data[:, :, :, i] = tmp
        return new_data.astype(np.float32)

    def get_relative_data(self, data_numpy):
        if self.data_type == 'relative':
            new_data = data_numpy
        elif self.data_type == 'multi_relative':
            anchor = [4, 8, 12, 16]  # 5,9,13,17
            C, T, V, M = data_numpy.shape
            new_data = np.zeros([C * 4, T, V, M])
            for i, anc in enumerate(anchor):
                center = data_numpy[:, :, anc, :].reshape([C, T, 1, M])
                new_data[i * 3:(i + 1) * 3, :, :, :] = data_numpy - center
        else:
            raise TypeError('Invalid data type: %s' % self.data_type)
        return new_data.astype(np.float32)

def get_action_class_name(fp):
    action_class = fp.split("_")[1]
    if action_class == "sitcut":
        action_class = "cut"
    if action_class == "standup":
        action_class = "stand"
    return action_class

single_person_classes = ["choke", "hang", "cut", "seizure", "sleep", "sit", "stand",
                         "throw", "pickup", "drink", "clap", "wave", "swing", "kick", "fall",
                         "sneeze", "nausea", "walk", "sitnm", "standnm"]
subject_train_subject_ids_ff = ["s01", "s03", "s04", "s05", "s09", "s11", "s13", "s15", "s017", "s19", "s21"]
subject_test_subject_ids_ff = ["s02", "s06", "s07", "s08", "s10", "s12", "s14", "s16", "s18", "s20"]



single_person_class_mapping = {k: v for k, v in enumerate(single_person_classes)}
single_person_mapping_reversed = {v: k for k, v in single_person_class_mapping.items()}

def filter_fps(all_fps, split, class_keys, outer_fp, with_join=True):
    filtered_fps = []
    for fp in all_fps:
        action_class = get_action_class_name(fp)
        if action_class not in class_keys:
            continue
        if split is not None:
            for s in split:
                if s in fp:
                    if with_join:
                        filtered_fps.append(os.path.join(outer_fp, fp))
                    else:
                        filtered_fps.append(fp)
                    break
        else: # all training samples that have the appropriate class keys
            if with_join:
                filtered_fps.append(os.path.join(outer_fp, fp))
            else:
                filtered_fps.append(fp)
    return filtered_fps

def noise_filter(data, fps, conf_val_thresh=0.5, missing_frames_thresh=0.4,
                 pad_ratio_thresh=0.4, swap_ratio_thresh=0.80):
    filtered_fps = []
    for fp in fps:
        num_people = len(data[fp])
        noise = data[fp][0]["noise"]
        conf_val_ratio = noise["conf_vals_r"]
        missing_frames_ratio = noise["missing_frames_r"]
        if num_people == 2:
            pad_ratio = noise["pad_ratio"]
            swap_ratio = noise["swap_count_ratio"]
        if conf_val_thresh is not None:
            if conf_val_ratio < conf_val_thresh:
                continue
        if missing_frames_ratio is not None:
            if missing_frames_ratio >= missing_frames_thresh:
                continue
        if pad_ratio_thresh is not None and num_people == 2:
            if pad_ratio > pad_ratio_thresh:
                continue
        if swap_ratio_thresh is not None and num_people == 2:
            if swap_ratio > swap_ratio_thresh:
                continue
        filtered_fps.append(fp)
    print(f"Filtered {len(fps)-len(filtered_fps)} noisy skeletal sequences.")
    return filtered_fps


class NTUMotionProcessorSpec(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
    """

    def __init__(self,
                 data_path,
                 data_type='normal',
                 t_length=300,
                 y_rotation=False,
                 displacement=False,
                 sampling='force_crop',
                 joint_num=32,
                 person_num=1,
                 part='val'):
        self.data_path = data_path
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.class_mapping = single_person_mapping_reversed
        class_keys = list(self.class_mapping.keys())

        all_fps = list(self.data.keys())

        if part == 'train':
            split = subject_train_subject_ids_ff
        elif part == 'val':
            split = subject_test_subject_ids_ff
        else:
            split = None

        self.fps = filter_fps(all_fps, split, class_keys, outer_fp=None, with_join=False)
        self.fps = list(filter(lambda x: "merged" not in x, self.fps))
        self.fps = noise_filter(self.data, self.fps)
        self.N, self.C, self.T, self.V, self.M = len(self.fps), 3, 300, joint_num, person_num

        self.data_type = data_type
        self.max_length = 300
        self.displacement = displacement
        self.sampling = sampling
        self.y_rotation = y_rotation
        self.t_length = t_length if t_length < self.max_length else self.max_length
        neighbor_1base = [(21, 2), (21, 3), (21, 5), (21, 9), (3, 4),
                          (5, 6), (6, 7), (7, 8), (8, 23), (23, 22),
                          (9, 10), (10, 11), (11, 12), (12, 25), (25, 24),
                          (2, 1), (1, 13), (1, 17), (13, 14), (14, 15),
                          (15, 16), (17, 18), (18, 19), (19, 20)]
        self.neighbor_1base = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        self.class_mapping = single_person_mapping_reversed


    def __len__(self):
        return len(self.fps)

    def __getitem__(self, index):
        # get data
        fp = self.fps[index]
        action_class_name = get_action_class_name(fp)
        label = self.class_mapping[action_class_name]
        body_data = self.data[fp].copy()
        num_bodies = len(body_data)
        # print(action_class_name)

        if action_class_name in single_person_classes and num_bodies == 2:
            body_data.pop(0)  # sorted by variance, pop the one with least variance at index 0
        if self.V == 32:
            data_numpy = np.expand_dims(body_data[0]["joint_values_32"], axis=0)
        if self.V == 25:
            data_numpy = np.expand_dims(body_data[0]["joint_values_25"], axis=0)

        assert data_numpy.shape[0] == 1

        data_numpy = data_numpy.transpose(3, 1, 2, 0)
        # data_numpy = np.array(self.data[index])
        # label = self.label[index]

        # get motion (displacement)
        if self.displacement > 0:
            motion_data = np.zeros_like(data_numpy)
            motion_data[:, :-self.displacement, :, :] = \
                data_numpy[:, self.displacement:, :, :] - data_numpy[:, :-self.displacement, :, :]
        else:
            motion_data = None

        # get skeleton sequence length, person number
        length = self.get_length(data_numpy)
        num = self.get_person_num(data_numpy)

        # relative coordinate
        if self.data_type == 'relative':
            # center = 21
            C, T, V, M = data_numpy.shape
            center = data_numpy[:, :, 2, :].reshape([C, T, 1, M])
            data_numpy = data_numpy - center
        elif self.data_type == 'normal':
            pass
        else:
            raise TypeError('Invalid data type: %s' % self.data_type)

        # crop length
        if self.sampling == 'force_crop':
            if self.displacement > 0:
                motion_data = motion_data[:, :self.t_length, :, :]
            data_numpy = data_numpy[:, :self.t_length, :, :]
        elif self.sampling == 'resize':
            data_numpy = self.real_resize(data_numpy, length, self.t_length)
            if motion_data is not None:
                motion_data = self.real_resize(motion_data, length, self.t_length)

        else:
            raise TypeError('Invalid sampling type: ' % self.sampling)

        # y rotation
        if self.y_rotation:
            for n in range(M):
                tmp_new, tmp_old = parallel_skeleton(data_numpy[:, :, :, n])
                data_numpy[:, :, :, n] = tmp_new.transpose(2, 0, 1)

        # get actual length
        length = self.t_length if length > self.t_length else length

        # return motion_data, label
        if motion_data is not None:
            return data_numpy, motion_data, label, fp
        else:
            return data_numpy, label, fp

    @staticmethod
    def get_length(data):
        length = (abs(data[:, :, 0, 0]).sum(axis=0) != 0).sum()
        return length

    @staticmethod
    def get_person_num(data):
        num = (abs(data[:, :, 0, :]).sum(axis=0).sum(axis=0) != 0) * 1.0
        return num

    @staticmethod
    def real_resize(data_numpy, length, crop_size):
        C, T, V, M = data_numpy.shape
        new_data = np.zeros([C, crop_size, V, M])
        for i in range(M):
            tmp = cv2.resize(data_numpy[:, :length, :, i].transpose(
                [1, 2, 0]), (V, crop_size), interpolation=cv2.INTER_LINEAR)
            tmp = tmp.transpose([2, 0, 1])
            new_data[:, :, :, i] = tmp
        return new_data.astype(np.float32)

    def get_relative_data(self, data_numpy):
        if self.data_type == 'relative':
            new_data = data_numpy
        elif self.data_type == 'multi_relative':
            anchor = [4, 8, 12, 16]  # 5,9,13,17
            C, T, V, M = data_numpy.shape
            new_data = np.zeros([C * 4, T, V, M])
            for i, anc in enumerate(anchor):
                center = data_numpy[:, :, anc, :].reshape([C, T, 1, M])
                new_data[i * 3:(i + 1) * 3, :, :, :] = data_numpy - center
        else:
            raise TypeError('Invalid data type: %s' % self.data_type)
        return new_data.astype(np.float32)
