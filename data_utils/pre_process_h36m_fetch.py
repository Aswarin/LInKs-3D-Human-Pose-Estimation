"""Use this file to preprocess the H36M Data obtained after
following the instructions in:
https://github.com/anibali/h36m-fetch
This produces a file which is able to be used by our h36m dataset class
"""

import h5py
import pickle
import os
import numpy as np

train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
test_subhects = ['S9', 'S11']

file_location = 'processed/'

joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27] #correct order
all_subjects_list = os.listdir(file_location)

processed_data = {}
for subject in all_subjects_list:
    all_actions_list = os.listdir(file_location + subject + '/')
    subjects_2d_pose_data = []
    subjects_3d_pose_data = []
    subjects_3d_univ_data = []
    for action in all_actions_list:
        anno_file = h5py.File(file_location + subject + '/' + action + '/annot.h5')
        pose_data = anno_file['pose']
        subjects_2d_pose_data.append(np.array(pose_data['2d'])[:, joints, :])
        subjects_3d_pose_data.append(np.array(pose_data['3d'])[:, joints, :])
        subjects_3d_univ_data.append(np.array(pose_data['3d-univ'])[:, joints, :])
    processed_data[subject] = {'poses_3d' : np.concatenate(subjects_3d_pose_data), 'poses_2d': np.concatenate(subjects_2d_pose_data), 'poses_3d_univ':np.concatenate(subjects_3d_univ_data)}

with open('h36m_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)
