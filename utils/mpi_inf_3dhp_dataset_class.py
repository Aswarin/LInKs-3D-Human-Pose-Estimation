import torch.nn
import torch.optim
from sklearn.decomposition import PCA
from lift_and_fill_models.utils.helpers import *
from torch.utils.data import Dataset
import pickle

class MPI_INF_3DHP_Dataset(Dataset):
    def __init__(self, file_name, train=False, joints=17, get_pca=False, normalize_func=None, get_2dgt=False, subjects=['S1', 'S2', 'S3', 'S4', 'S5', 'S6']):
        self.train = train
        self.data = dict()
        self.get_2dgt = get_2dgt
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        two_d_joints = []
        three_d_joints = []
        for s in subjects:
            two_d_joints.append(data[s]['poses_2d'])
            three_d_joints.append(data[s]['poses_3d_univ'])
        two_d_joints = np.concatenate(two_d_joints)
        three_d_joints = np.concatenate(three_d_joints)
        three_d_joints = three_d_joints.transpose(0, 2, 1).reshape(-1, 3 * joints)
        self.data['poses_3d'] = three_d_joints
        if normalize_func:
            two_d_joints = two_d_joints.transpose(0, 2, 1).reshape(-1, 2 * joints)
            two_d_joints = normalize_func(two_d_joints)
            self.data['poses_2d'] = two_d_joints.astype(np.float32)
        else:
            scale = list()
            norm_pose = list()
            for t in range(len(two_d_joints)):
                keypoints = two_d_joints[t]
                keypoints = keypoints - keypoints[0]
                pose_max = np.max(abs(keypoints))
                normalised_two_d_keypoints = keypoints / pose_max
                scale.append(pose_max)
                normalised_two_d_keypoints = normalised_two_d_keypoints.transpose(1, 0).reshape(-1, 2 * joints)
                norm_pose.append(normalised_two_d_keypoints)
            norm_pose = np.concatenate(norm_pose)

            self.data['poses_2d'] = norm_pose

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