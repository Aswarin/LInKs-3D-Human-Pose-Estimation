import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from torch.utils.data import Dataset
from matplotlib import animation
from time import time
from types import SimpleNamespace
import torch.optim
# import data
from torch.utils import data
import torch.nn
import torch.optim
from sklearn.decomposition import PCA
from lift_and_fill_models.utils.helpers import *
from torch.utils.data import Dataset
import pickle
import argparse

pose_choice = 7
class H36M_Data(Dataset):
    def __init__(self, file_name, train=False, joints=17, get_pca=False, normalize_func=None, get_2dgt=False, subjects=['S1', 'S5', 'S7', 'S6', 'S8']):
        self.train = train
        self.data = dict()
        self.get_2dgt = get_2dgt
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        two_d_joints = []
        three_d_joints = []
        for s in subjects:
            two_d_joints.append(data[s]['poses_2d'])
            three_d_joints.append(data[s]['poses_3d'])
        two_d_joints = np.concatenate(two_d_joints)
        three_d_joints = np.concatenate(three_d_joints)
        three_d_joints = three_d_joints.transpose(0, 2, 1).reshape(-1, 3 * joints)
        self.data['poses_3d'] = three_d_joints
        if normalize_func:
            two_d_joints = two_d_joints.transpose(0, 2, 1).reshape(-1, 2 * joints)
            two_d_joints = normalize_func(two_d_joints)
            self.data['poses_2d'] = two_d_joints
        else:
            two_d_joints = two_d_joints.transpose(0, 2, 1).reshape(-1, 2 * joints)
            self.data['poses_2d'] = two_d_joints

        if get_pca:
            self.pca = PCA()
            self.pca.fit(self.data['poses_2d'])


    def __len__(self):
        return self.data['poses_3d'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        if self.get_2dgt:
            sample['p2d_gt'] = self.data['poses_2d'][idx]
        else:
            sample['p2d_pred'] = self.data['poses_2d'][idx]

        sample['poses_3d'] = self.data['poses_3d'][idx]

        return sample

datafile = '../../EVAL_DATA/correct_interesting_frames_h36m.pkl'

# train_data = H36M_Data(datafile, train=True, get_pca=True, normalize_func=normalize_head, get_2dgt=True,
#                                  subjects=['S1', 'S5', 'S7', 'S6', 'S8'])
# pca = train_data.pca
test_data = H36M_Data(datafile, train=False, normalize_func=normalize_head_test, get_2dgt=True,
                                subjects=['S9', 'S11'])


predicted_poses = test_data.data['poses_3d'].reshape(-1, 3, 17)
predicted_poses = predicted_poses - predicted_poses[:, :, [0]]



predicted_poses = predicted_poses.transpose(0, 2, 1)

pose = np.array(predicted_poses[pose_choice])
check = pose
buff_large = np.zeros((32, 3))
buff_large[(0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27), :] = pose
pose = buff_large.transpose()
#
kin = np.array([[0, 12], [12, 13], [13, 14], [15, 14], [13, 17], [17, 18], [18, 19], [13, 25], [25, 26], [26, 27], [0, 1], [1, 2], [2, 3], [0, 6], [6, 7], [7, 8]])

order = np.array([0, 2, 1])

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.view_init(azim=-45, elev=15)

for link in kin:
    ax.plot(pose[0, link], pose[2, link], -pose[1, link], linewidth=5.0)

ax.legend()
# ax.set_xlabel('X')
# ax.set_ylabel('Z')
# ax.set_zlabel('Y')
ax.set_aspect('auto')
#ax.axis('off')

X = pose[0, :]
Y = pose[2, :]
Z = -pose[1, :]
max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

mid_x = (X.max() + X.min()) * 0.5
mid_y = (Y.max() + Y.min()) * 0.5
mid_z = (Z.max() + Z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.show()