import random
import numpy as np
import pickle, torch
import feeder.tools as tools
import os
# from feeder.SkelVisualizer import visualize_skel


PKU_classes = ["bow", "brushing hair", "brushing teeth", "check time (from watch)",	"cheer up",	"clapping",
               "cross hands", "drink water",	"drop",	"eat meal/snack", "falling",
               "giving something to other person", "hand waving", "handshaking",	"hopping (one foot jumping)",
               "hugging other person",	"jump up", "kicking", "kicking something",
               "make a phone call/answer phone",
               "pat on back of other person", "pickup",	"playing with phone/tablet", "point finger",
               "point to sth.",	"punching/slapping other person", "pushing other person",
               "put on a hat",	"put sth. in pkt.",	"reading",	"rub hands",	"salute",
               "sitting down",	"standing up",	"take off a hat",	"take off glasses",	"take off jacket",
               "take out something from pocket", "taking a selfie",	"tear up paper", "throw", "touch back",
               "touch chest",	"touch head", "touch neck",
               "typing", "use a fan (with hand or paper)/feeling warm",
               "wear jacket", "wear on glasses", "wipe face", "writing"]

PKU_class_mapping = {k: v for k, v in enumerate(PKU_classes)}
PKU_class_mapping_reversed = {v: k for k, v in PKU_class_mapping.items()}

NTU = {1: "drink water", 2: "eat meal/snack", 3: "brushing teeth",4: "brushing hair", 5:"drop", 6: "pickup", 7: "throw",
8: "sitting down",9: "standing up (from sitting position)",10: "clapping",11: "reading", 12: "writing", 13: "tear up paper",
14: "wear jacket",15: "take off jacket",16: "wear a shoe",17: "take off a shoe", 18: "wear on glasses", 19: "take off glasses",
20: "put on a hat/cap",21: "take off a hat/cap",22: "cheer up",23: "hand waving", 24: "kicking something", 25: "reach into pocket",
26: "hopping (one foot jumping)",27: "jump up",28: "make a phone call/answer phone", 29: "playing with phone/tablet",
30: "typing on a keyboard",31: "pointing to something with finger",32: "taking a selfie",33: "check time (from watch)",
34: "rub two hands together",35: "nod head/bow",36: "shake head",37: "wipe face",38: "salute",39: "put the palms together",
40: "cross hands in front (say stop)",41: "sneeze/cough",42: "staggering",43: "falling",44: "touch head (headache)",
45: "touch chest (stomachache/heart pain",46: "touch back (backache)",47: "touch neck (neckache)",48: "nausea or vomiting condition",
49: "use a fan (with hand or paper)/feeling warm",50: "punching/slapping other person",51: "kicking other person",
52: "pushing other person",53: "pat on back of other person",54: "point finger at the other person",55: "hugging other person",
56: "giving something to other person", 57: "touch other person's pocket", 58: "handshaking",59: "walking towards each other",
60: "walking apart from each other"}

key_maping_from_NTU60ToPKU = {'34': 0, '3': 1, '2': 2, '32': 3, '21': 4, '9': 5, '39': 6, '0': 7, '4': 8, '1': 9,
                            '42': 10, '55': 11, '22': 12, '57': 13, '25': 14, '54': 15, '26': 16, '50': 17, '23': 18,
                            '27': 19, '52': 20, '5': 21, '28': 22, '53': 23, '30': 24, '49': 25, '51': 26, '19': 27, '24': 28,
                            '10': 29, '33': 30, '37': 31, '7': 32, '8': 33, '20': 34, '18': 35, '14': 36, '-100': 37, '31': 38,
                            '12': 39, '6': 40, '45': 41, '44': 42, '43': 43, '46': 44, '29': 45, '48': 46, '13': 47,
                            '17': 48, '36': 49, '11': 50}  # start from '0', which means '34' is '35' in NTU above
                                                           # note that there is a map "'-100': 37", so this mapping dict has 51 actions

different_actions = [15, 16, 35, 38, 40, 41, 47, 56, 58, 59] # not used if transfer NTU to PKU

values_mapping_from_NTU60ToPKU = {'nod head/bow': 'bow', 'brushing hair': 'brushing hair', 'brushing teeth': 'brushing teeth',
      'check time (from watch)': 'check time (from watch)', 'cheer up': 'cheer up',
      'clapping': 'clapping', 'cross hands in front (say stop)': 'cross hands in front (say stop)',
      'drink water': 'drink water', 'drop': 'drop', 'eat meal/snack': 'eat meal/snack',
      'falling': 'falling', 'giving something to other person': 'giving something to other person',
      'hand waving': 'hand waving', 'handshaking': 'handshaking',
      'hopping (one foot jumping)': 'hopping (one foot jumping)', 'hugging other person': 'hugging other person',
      'jump up': 'jump up', 'kicking other person': 'kicking other person', 'kicking something': 'kicking something',
      'make a phone call/answer phone': 'make a phone call/answer phone',
      'pat on back of other person': 'pat on back of other person', 'pickup': 'pickup',
      'playing with phone/tablet': 'playing with phone/tablet',
      'point finger at the other person': 'point finger at the other person',
      'pointing to something with finger': 'pointing to something with finger',
      'punching/slapping other person': 'punching/slapping other person',
      'pushing other person': 'pushing other person', 'put on a hat/cap': 'put on a hat/cap',
      'reading': 'reading', 'rub two hands together': 'rub two hands together', 'salute': 'salute',
      'sitting down': 'sitting down', 'standing up (from sitting position)': 'standing up',
      'take off a hat/cap': 'take off a hat/cap', 'take off glasses': 'take off glasses',
      'take off jacket': 'take off jacket', 'taking a selfie': 'taking a selfie',
      'tear up paper': 'tear up paper', 'throw': 'throw', 'touch back (backache)': 'touch back (backache)',
      'touch chest (stomachache/heart pain': 'touch chest (stomachache/heart pain)',
      'touch head (headache)': 'touch head (headache)', 'touch neck (neckache)': 'touch neck (neckache)',
      'typing on a keyboard': 'typing on a keyboard',
      'use a fan (with hand or paper)/feeling warm': 'use a fan (with hand or paper)/feeling warm',
      'wear jacket': 'wear jacket', 'wear on glasses': 'wear on glasses', 'wipe face': 'wipe face', 'writing': 'writing'}

class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        if 'ntu' in self.data_path:
            labels_mapped_to_PKU = []
            for i in self.label:
                mapped_label = key_maping_from_NTU60ToPKU[str(i)] if str(i) in list(
                    key_maping_from_NTU60ToPKU.keys()) else -100
                labels_mapped_to_PKU.append(mapped_label)
            self.label = labels_mapped_to_PKU  # 51 actions after mapping (including a '-100' action)
            print('###################51 actions after mapping (including a -100 action)##########################')


        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy





class Feeder_triple(torch.utils.data.Dataset):
    """ Feeder for triple inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True,
                 aug_method='12345'):
        self.data_path = data_path
        self.label_path = label_path
        self.aug_method = aug_method

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)


        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # reduce the dataset to have the same label space with PKU
        index = [i for i, n in enumerate(self.sample_name) if int(n[-11:-9]) not in different_actions]
        self.data = self.data[index]
        self.sample_name = [self.sample_name[i] for i in index]
        self.label = [self.label[i] for i in index]
        print("#################################using reduced data for transfer learning#################################")

        # if 'train' in self.label_path:
        #     self.sample_name, self.label, self.data = self.sample_name[::4], self.label[::4], self.data[::4]
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data1 = self._strong_aug(data_numpy)
        data2 = self._aug(data_numpy)
        data3 = self._aug(data_numpy)
        return [data1, data2, data3], label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        return data_numpy


    # you can choose different combinations
    def _strong_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)

        return data_numpy


class Feeder_semi(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, label_percent=0.1, shear_amplitude=0.5, temperal_padding_ratio=6,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.label_percent = label_percent

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        n = len(self.label)
        # Record each class sample id
        class_blance = {}
        for i in range(n):
            if self.label[i] not in class_blance:
                class_blance[self.label[i]] = [i]
            else:
                class_blance[self.label[i]] += [i]

        final_choise = []
        for c in class_blance:
            c_num = len(class_blance[c])
            choise = random.sample(class_blance[c], round(self.label_percent * c_num))
            final_choise += choise
        final_choise.sort()

        self.data = self.data[final_choise]
        new_sample_name = []
        new_label = []
        for i in final_choise:
            new_sample_name.append(self.sample_name[i])
            new_label.append(self.label[i])

        self.sample_name = new_sample_name
        self.label = new_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy

class Feeder_mixed_withNTU(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, label_percent=0.1, shear_amplitude=0.5, temperal_padding_ratio=6,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path
        self.data_path_ntu = './data/gty/action_dataset/ntu60_frame50/xsub/train_position.npy'
        self.label_path_ntu = './data/gty/action_dataset/ntu60_frame50/xsub/train_label.pkl'
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.label_percent = label_percent

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        with open(self.label_path_ntu, 'rb') as f:
            sample_name_ntu, label_ntu = pickle.load(f)
        labels_mapped_to_PKU = []
        for i in self.label:
            mapped_label = key_maping_from_NTU60ToPKU[str(i)] if str(i) in list(
                key_maping_from_NTU60ToPKU.keys()) else -100
            labels_mapped_to_PKU.append(mapped_label)
        label_ntu = labels_mapped_to_PKU  # 51 actions after mapping (including a '-100' action)
        print('###################adding NTU data##########################')
        # load data
        if mmap:
            data_ntu = np.load(self.data_path_ntu, mmap_mode='r')
        else:
            data_ntu = np.load(self.data_path_ntu)

        n = len(self.label)
        # Record each class sample id
        class_blance = {}
        for i in range(n):
            if self.label[i] not in class_blance:
                class_blance[self.label[i]] = [i]
            else:
                class_blance[self.label[i]] += [i]

        final_choise = []
        for c in class_blance:
            c_num = len(class_blance[c])
            choise = random.sample(class_blance[c], round(self.label_percent * c_num))
            final_choise += choise
        final_choise.sort()

        self.data = np.concatenate((self.data[final_choise], data_ntu))

        new_sample_name = []
        new_label = []
        for i in final_choise:
            new_sample_name.append(self.sample_name[i])
            new_label.append(self.label[i])

        self.sample_name = new_sample_name
        self.sample_name.extend(sample_name_ntu)
        self.label = new_label
        self.label.extend(label_ntu)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy


if __name__ == '__main__':
    # key_maping_from_NTU60ToPKU = {}
    # values_mapping_from_NTU60ToPKU = {}
    # for i, (_, p) in enumerate(PKU_class_mapping.items()):
    #     for j, (k, v) in enumerate(NTU.items()):
    #         if p in v or v in p:
    #             key_maping_from_NTU60ToPKU[str(j)] = i
    #             values_mapping_from_NTU60ToPKU[v] = p
    #
    debug = 'debug'
    data_path = r'K:\AimCLR-main\data/gty/action_dataset/ntu60_frame50/xsub/val_position.npy'
    label_path = r'K:\AimCLR-main\data\gty/action_dataset/ntu60_frame50/xsub/val_label.pkl'

    dataloader = Feeder_single(data_path=data_path, label_path=label_path, shear_amplitude=-1, temperal_padding_ratio=-1)
    dataloader.vis_skel_vids()




