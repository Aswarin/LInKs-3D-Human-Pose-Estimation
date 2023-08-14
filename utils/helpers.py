import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

def combine_left_right_pred_3d(left_split, right_split, choice):
    left_split = left_split.reshape(-1, 3, 11)
    right_split = right_split.reshape(-1, 3, 11)
    if choice == 'right':
        full_pose = torch.stack((right_split[:, :, 0], right_split[:, :, 1], right_split[:, :, 2], right_split[:, :, 3],
                                 left_split[:, :, 1], left_split[:, :, 2], left_split[:, :, 3], right_split[:, :, 4],
                                 right_split[:, :, 5], right_split[:, :, 6], right_split[:, :, 7], left_split[:, :, 8], left_split[:, :, 9],
                                 left_split[:, :, 10], right_split[:, :, 8], right_split[:, :, 9], right_split[:, :, 10]), dim=2)
    else:
        full_pose = torch.stack((left_split[:, :, 0], right_split[:, :, 1], right_split[:, :, 2], right_split[:, :, 3],
                                 left_split[:, :, 1], left_split[:, :, 2], left_split[:, :, 3], left_split[:, :, 4],
                                 left_split[:, :, 5], left_split[:, :, 6], left_split[:, :, 7], left_split[:, :, 8], left_split[:, :, 9],
                                 left_split[:, :, 10], right_split[:, :, 8], right_split[:, :, 9], right_split[:, :, 10]), dim=2)
    return full_pose.reshape(-1, 51)


def combine_left_right_pred_2d(left_split, right_split, choice):
    left_split = left_split.reshape(-1, 2, 11)
    right_split = right_split.reshape(-1, 2, 11)
    if choice == 'right':
        full_pose = torch.stack((right_split[:, :, 0], right_split[:, :, 1], right_split[:, :, 2], right_split[:, :, 3],
                                 left_split[:, :, 1], left_split[:, :, 2], left_split[:, :, 3], right_split[:, :, 4],
                                 right_split[:, :, 5], right_split[:, :, 6], right_split[:, :, 7], left_split[:, :, 8], left_split[:, :, 9],
                                 left_split[:, :, 10], right_split[:, :, 8], right_split[:, :, 9], right_split[:, :, 10]), dim=2)
    else:
        full_pose = torch.stack((left_split[:, :, 0], right_split[:, :, 1], right_split[:, :, 2], right_split[:, :, 3],
                                 left_split[:, :, 1], left_split[:, :, 2], left_split[:, :, 3], left_split[:, :, 4],
                                 left_split[:, :, 5], left_split[:, :, 6], left_split[:, :, 7], left_split[:, :, 8], left_split[:, :, 9],
                                 left_split[:, :, 10], right_split[:, :, 8], right_split[:, :, 9], right_split[:, :, 10]), dim=2)


    return full_pose.reshape(-1, 34)

def combine_left_right_pred_1d(left_split, right_split, choice):
    left_split = left_split.reshape(-1, 1, 11)
    right_split = right_split.reshape(-1, 1, 11)
    if choice == 'right':
        combined_depth = torch.stack((right_split[:, :, 0], right_split[:, :, 1], right_split[:, :, 2], right_split[:, :, 3],
                                 left_split[:, :, 1], left_split[:, :, 2], left_split[:, :, 3], right_split[:, :, 4],
                                 right_split[:, :, 5], right_split[:, :, 6], right_split[:, :, 7], left_split[:, :, 8], left_split[:, :, 9],
                                 left_split[:, :, 10], right_split[:, :, 8], right_split[:, :, 9], right_split[:, :, 10]), dim=2)
    else:
        combined_depth = torch.stack((left_split[:, :, 0], right_split[:, :, 1], right_split[:, :, 2], right_split[:, :, 3],
                                 left_split[:, :, 1], left_split[:, :, 2], left_split[:, :, 3], left_split[:, :, 4],
                                 left_split[:, :, 5], left_split[:, :, 6], left_split[:, :, 7], left_split[:, :, 8], left_split[:, :, 9],
                                 left_split[:, :, 10], right_split[:, :, 8], right_split[:, :, 9], right_split[:, :, 10]), dim=2)
    return combined_depth

def split_data_left_right(data):
    data = data.reshape(-1, 2, 17)
    right = torch.stack((data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3],
                        data[:, :, 7], data[:, :, 8], data[:, :, 9], data[:, :, 10],
                         data[:, :, 14], data[:, :, 15], data[:, :, 16]), dim=2).reshape(-1, 22)

    left = torch.stack((data[:, :, 0], data[:, :, 4], data[:, :, 5], data[:, :, 6], data[:, :, 7],
                        data[:, :, 8], data[:, :, 9], data[:, :, 10], data[:, :, 11],
                        data[:, :, 12], data[:, :, 13]), dim=2).reshape(-1, 22)

    return left, right


def split_data_left_right_v2(data):
    data = data.reshape(-1, 2, 17)
    right = torch.stack((data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3],
                        data[:, :, 7], data[:, :, 8], data[:, :, 9], data[:, :, 10],
                         data[:, :, 11], data[:, :, 12], data[:, :, 13]), dim=2).reshape(-1, 22)

    left = torch.stack((data[:, :, 0], data[:, :, 4], data[:, :, 5], data[:, :, 6], data[:, :, 7],
                        data[:, :, 8], data[:, :, 9], data[:, :, 10], data[:, :, 14],
                        data[:, :, 15], data[:, :, 16]), dim=2).reshape(-1, 22)

    return left, right


def split_data_left_right_3d(data):
    data = data.reshape(-1, 2, 17)
    right = torch.stack((data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3],
                        data[:, :, 7], data[:, :, 8], data[:, :, 9], data[:, :, 10],
                         data[:, :, 14], data[:, :, 15], data[:, :, 16]), dim=2).reshape(-1, 33)

    left = torch.stack((data[:, :, 0], data[:, :, 4], data[:, :, 5], data[:, :, 6], data[:, :, 7],
                        data[:, :, 8], data[:, :, 9], data[:, :, 10], data[:, :, 11],
                        data[:, :, 12], data[:, :, 13]), dim=2).reshape(-1, 33)

    return left, right


def split_data_left_right_numpy(data):
    pose_data = data.reshape(-1, 2, 17)
    right = np.stack((pose_data[:, :, 0], pose_data[:, :, 1], pose_data[:, :, 2], pose_data[:, :, 3],
                        pose_data[:, :, 7], pose_data[:, :, 8], pose_data[:, :, 9], pose_data[:, :, 10],
                         pose_data[:, :, 14], pose_data[:, :, 15], pose_data[:, :, 16]), axis=2).reshape(-1, 22)

    left = np.stack((pose_data[:, :, 0], pose_data[:, :, 4], pose_data[:, :, 5], pose_data[:, :, 6], pose_data[:, :, 7],
                        pose_data[:, :, 8], pose_data[:, :, 9], pose_data[:, :, 10], pose_data[:, :, 11],
                        pose_data[:, :, 12], pose_data[:, :, 13]), axis=2).reshape(-1, 22)

    return left, right

def temporal_split_data_left_right(data):
    pose_data = data.reshape(-1, 2, 2, 17)
    right = torch.stack((pose_data[:, :, :, 0], pose_data[:, :, :, 1], pose_data[:, :, :, 2], pose_data[:, :, :, 3],
                        pose_data[:, :, :, 7], pose_data[:, :, :, 8], pose_data[:, :, :, 9], pose_data[:, :, :, 10],
                         pose_data[:, :, :, 14], pose_data[:, :, :, 15], pose_data[:, :, :, 16]), dim=3).reshape(-1, 44)

    left = torch.stack((pose_data[:, :, :, 0], pose_data[:, :, :, 4], pose_data[:, :, :, 5], pose_data[:, :, :, 6], pose_data[:, :, :, 7],
                        pose_data[:, :, :, 8], pose_data[:, :, :, 9], pose_data[:, :, :, 10], pose_data[:, :, :, 11],
                        pose_data[:, :, :, 12], pose_data[:, :, :, 13]), dim=3).reshape(-1, 44)

    return left, right




def combine_left_right_occluded_3d(occluded_part, visible_part, part_occluded):
    occluded_part = occluded_part.reshape(-1, 3, 6)
    visible_part = visible_part.reshape(-1, 3, 11)
    if part_occluded == 'right':
        full_pose = torch.stack((visible_part[:, :, 0], occluded_part[:, :, 0], occluded_part[:, :, 1], occluded_part[:, :, 2],
                                 visible_part[:, :, 1], visible_part[:, :, 2], visible_part[:, :, 3], visible_part[:, :, 4],
                                 visible_part[:, :, 5], visible_part[:, :, 6], visible_part[:, :, 7], visible_part[:, :, 8], visible_part[:, :, 9],
                                 visible_part[:, :, 10], occluded_part[:, :, 3], occluded_part[:, :, 4], occluded_part[:, :, 5]), dim=2)
    else:
        full_pose = torch.stack((visible_part[:, :, 0], visible_part[:, :, 1], visible_part[:, :, 2], visible_part[:, :, 3],
                                 occluded_part[:, :, 0], occluded_part[:, :, 1], occluded_part[:, :, 2], visible_part[:, :, 4],
                                 visible_part[:, :, 5], visible_part[:, :, 6], visible_part[:, :, 7], occluded_part[:, :, 3], occluded_part[:, :, 4],
                                 occluded_part[:, :, 5], visible_part[:, :, 8], visible_part[:, :, 9], visible_part[:, :, 10]), dim=2)


    return full_pose


def get_bone_lengths_all(poses):
    bone_map = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
                [12, 13], [8, 14], [14, 15], [15, 16]]

    poses = poses.reshape((-1, 3, 17))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths

def get_bone_lengths_legs(poses):
    bone_map = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]]

    poses = poses.reshape((-1, 3, 7))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths

def get_bone_lengths_torso(poses):
    poses = poses.reshape(-1, 3, 10)
    root = torch.zeros(poses.shape[0], 3, 1).cuda()
    poses = torch.concatenate((root, poses), dim=2)
    bone_map = [[0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [5, 6],
                [6, 7], [2, 8], [8, 9], [9, 10]]

    poses = poses.reshape((-1, 3, 11))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths
# 7 =4, 8 = 5 etc
def get_bone_lengths_left_right(poses):
    bone_map = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [6, 7], [5, 8], [8, 9],
                [9, 10]]

    poses = poses.reshape((-1, 3, 11))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths


def normalize_head(poses_2d, root_joint=0):
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [root_joint]]

    scale = np.linalg.norm(p2d[:, :, 0] - p2d[:, :, 10], axis=1, keepdims=True)
    p2ds = poses_2d / scale.mean()
    p2ds = p2ds * (1 / 10)

    return p2ds

def interpolate_gaussian_batch(latent_variables, t):
    #Interpolate between pairs of Gaussian latent variables.

    if len(latent_variables) % 2 != 0:
        raise ValueError("Batch size must be even for interpolation.")
    # Reshape the batch into pairs for interpolation.
    pairs = latent_variables.reshape(-1, 2, 34)

    # Perform element-wise linear interpolation between each pair of latent variables.
    interpolated_poses = (1 - t) * pairs[:, 0] + t * pairs[:, 1]

    return interpolated_poses

def normalize_head_test(poses_2d, scale=145.40964): #ground truth in training scale=145.5329587164913, MPI = 324.8037559356081, 142.34154 for all frames 145.40964 for interesting)
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [0]]

    p2ds = poses_2d / scale
    p2ds = p2ds * (1 / 10)

    return p2ds

def normalize_head_test_mpi_chest(poses_2d, scale=318.79249520730474): #655.3278874377231 #318.79249520730474 #369.93078334167893 for mpi test, ground truth in training scale=145.5329587164913, MPI = 320.75802546017553, 142.34154 for all frames 145.40964 for interesting)
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [0]]
    p2ds = poses_2d / scale
    p2ds = p2ds * (1 / 10)

    return p2ds


def normalize_head_test_mpi_vnect(poses_2d, scale=302.8530630720979): #655.3278874377231 #318.79249520730474 #369.93078334167893 for mpi test, ground truth in training scale=145.5329587164913, MPI = 320.75802546017553, 142.34154 for all frames 145.40964 for interesting)
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [0]]
    p2ds = poses_2d / scale
    p2ds = p2ds * (1 / 10)

    return p2ds

def normalize_head_test_temporal(poses_2d, scale=145.40419): #ground truth in training scale=145.5329587164913)
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 17)
    p2d -= p2d[:, :, [0]]

    p2ds = poses_2d / scale
    p2ds = p2ds * (1 / 10)

    return p2ds


def perspective_projection(pose_3d):
    pose_3d = pose_3d.reshape(-1, 51)
    p2d = pose_3d[:, 0:34].reshape(-1, 2, 17)
    p2d = p2d / pose_3d[:, 34:51].reshape(-1, 1, 17)

    return p2d.reshape(-1, 34)

def perspective_projection_legs(pose_3d):
    pose_3d = pose_3d.reshape(-1, 21)
    p2d = pose_3d[:, 0:14].reshape(-1, 2, 7)
    p2d = p2d / pose_3d[:, 14:21].reshape(-1, 1, 7)

    return p2d.reshape(-1, 14)

def perspective_projection_torso(pose_3d):
    pose_3d = pose_3d.reshape(-1, 30)
    p2d = pose_3d[:, 0:20].reshape(-1, 2, 10)
    p2d = p2d / pose_3d[:, 20:30].reshape(-1, 1, 10)

    return p2d.reshape(-1, 20)

def perspective_projection_left_right(pose_3d):
    pose_3d = pose_3d.reshape(-1, 33)
    p2d = pose_3d[:, 0:22].reshape(-1, 2, 11)
    p2d = p2d / pose_3d[:, 22:33].reshape(-1, 1, 11)

    return p2d.reshape(-1, 22)


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 1024), nn.ReLU(),
                         nn.Linear(1024, dims_out))




def add_noise(latent_vars, noise_factor):
    # Generate random noise of the same shape as the latent variables
    noise = torch.randn_like(latent_vars)

    # Compute the noise factor relative to the magnitudes of the latent variables
    noise_magnitudes = noise * latent_vars

    # Scale the noise by the noise factor and add it to the latent variables
    noisy_latent_vars = latent_vars + (noise_factor * noise_magnitudes)

    return noisy_latent_vars

import random


def occlusion_create(poses_2d):
    occluded_poses = poses_2d.clone()
    occluded_poses = occluded_poses.reshape(-1, 2, 17)
    #occlusion_maps = []

    for i in range(len(occluded_poses)):

        #limb_to_occlude = random.choice([['left_leg'], ['right_leg'], ['left_arm'], ['right_arm']])
        limb_to_occlude = random.choice([['left_leg']])

        if 'left_leg' in limb_to_occlude:
            keypoints_to_occlude = random.choice([[6], [5,6], [4,5,6]])
        elif 'right_leg' in limb_to_occlude:
            keypoints_to_occlude = random.choice([[3], [2,3], [1,2,3]])
        elif 'left_arm' in limb_to_occlude:
            keypoints_to_occlude = random.choice([[11], [11, 12], [11, 12, 13]])
        else:
            keypoints_to_occlude = random.choice([[14], [14,15], [14,15,16]])

        #occlusion_map = [1 if kp in keypoints_to_occlude else 0 for kp in range(occluded_poses.shape[2])]
        #occlusion_maps.append(occlusion_map)

        for kp in keypoints_to_occlude:
            occluded_poses[i, :, kp] = 0.0

    #occlusion_maps = torch.Tensor(occlusion_maps).reshape(-1, 1, 17)

    return occluded_poses.reshape(-1, 34)


