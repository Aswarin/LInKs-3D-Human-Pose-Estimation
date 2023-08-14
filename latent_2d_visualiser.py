"""Use this file to visualise the 2D poses drawn from the latent distribution of the normalising flow"""
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from utils.h36m_dataset_class import H36M_Data, MPI_INF_3DHP_Dataset
import torch.optim
from utils.helpers import *
import FrEIA.framework as Ff
import FrEIA.modules as Fm

pose_choice = 740

#change datafile as need be
datafile = '../EVAL_DATA/mpi_inf_data.pkl'

## load pretrained INN
# a simple chain of operations is collected by ReversibleSequential
left_inn2d = Ff.SequenceINN(22)
right_inn2d = Ff.SequenceINN(22)
full_inn2d = Ff.SequenceINN(34)
torso_inn2d = Ff.SequenceINN(20)
legs_inn2d = Ff.SequenceINN(14)
for k in range(8):
    left_inn2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    right_inn2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    full_inn2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    torso_inn2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    legs_inn2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
#
# left_inn2d.load_state_dict(torch.load('models/mpi_norm_flow_left_2.pt'))
# right_inn2d.load_state_dict(torch.load('models/mpi_norm_flow_right_2.pt'))
full_inn2d.load_state_dict(torch.load('models/mpi_norm_flow_sampling.pt'))
torso_inn2d.load_state_dict(torch.load('models/mpi_norm_flow_torso_2.pt'))
legs_inn2d.load_state_dict(torch.load('models/mpi_norm_flow_legs_2.pt'))
# freeze all weights in INN



# train_data = H36M_Data(datafile, train=True, normalize_func=normalize_head, get_2dgt=True,
#                                  subjects=['S1', 'S5', 'S7', 'S6', 'S8'])

# test_data = H36M_Data(datafile, train=False, normalize_func=normalize_head_test, get_2dgt=True,
#                                 subjects=['S9', 'S11'])

test_data = MPI_INF_3DHP_Dataset(datafile, train=False, normalize_func=normalize_head_test_mpi_vnect, get_2dgt=True,
                                subjects=['S1', 'S2'])


predicted_poses = torch.tensor(test_data.data['poses_2d'])

leg = predicted_poses.reshape(-1, 2, 17)[:, :, :7].reshape(-1, 14)
torso = predicted_poses.reshape(-1, 2, 17)[:, :, 7:].reshape(-1, 20)

torso_gaussian , _ = torso_inn2d(torso)
gaussian_noisy = add_noise(torso, noise_factor=0.2)
torso_samples, _ = torso_inn2d(torso_gaussian, rev=True)

drawn_samples = torch.concat((leg.reshape(-1, 2, 7), torso_samples.reshape(-1, 2, 10)), dim=2)

# left, right = split_data_left_right(predicted_poses)
#
#
# batch, _ = left_inn2d(left)
#
# batch_r, _ = right_inn2d(right)
#
# gaussian_noisy = add_noise(batch, noise_factor=0.2) #comment for our sampling
# gaussian_noisy_r = add_noise(batch_r, noise_factor=0.2) #comment for our sampling
# batch, _ = full_inn2d(predicted_poses)
# gaussian_noisy = add_noise(batch, noise_factor=0.2) #comment for our sampling
#gaussian_noisy = torch.randn_like(batch) #comment for random sampling


# drawn_samples, _ = full_inn2d(gaussian_noisy, rev=True)
drawn_samples = drawn_samples.reshape(-1, 2, 17)
drawn_samples[:, :, [0]] = 0.0
drawn_samples = drawn_samples.reshape(-1, predicted_poses.shape[1])
inp_samples = drawn_samples.data.cpu().numpy()


gt_pose = predicted_poses[pose_choice].cpu().numpy().reshape(-1, 2, 17)
fake_pose = inp_samples[pose_choice].reshape(-1, 2, 17)
fake_pose = fake_pose.transpose(0, 2, 1)
gt_pose = gt_pose.transpose(0, 2, 1)
buff_large_gt = np.zeros((32, 2))


buff_large_gt[(0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27), :] = gt_pose
gt_pose = buff_large_gt.transpose()

buff_large_fake = np.zeros((32, 2))
buff_large_fake[(0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27), :] = fake_pose
fake_pose = buff_large_fake.transpose()


kin = np.array([[0, 12], [12, 13], [13, 14], [15, 14], [13, 17], [17, 18], [18, 19], [13, 25], [25, 26], [26, 27],
                [0, 1], [1, 2], [2, 3], [0, 6], [6, 7], [7, 8]])

# kin = np.array([[0, 12], [12, 13], [13, 14], [15, 14],[13, 17], [17, 18], [18, 19]
#                ]) #missing arm

# kin = np.array([[0, 12], [12, 13], [13, 14], [15, 14], [13, 25], [25, 26], [26, 27],
#                 [0, 6], [6, 7], [7, 8]]) #missing arm



order = np.array([0, 2, 1])

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure(figsize=(6,6))

ax = fig.gca()
#ax.view_init(azim=-45, elev=15)

for link in kin:
    ax.plot(gt_pose[0, link], -gt_pose[1, link], linewidth=5.0, color='green')
    ax.plot(fake_pose[0, link], -fake_pose[1, link], linewidth=5.0, color='red')

ax.legend()
ax.set_aspect('auto')
#plt.title('Occluded 2D Pose', fontsize=16, y=1.05)

X = gt_pose[0, :]
Z = -gt_pose[1, :]
max_range = np.array([X.max() - X.min(), Z.max() - Z.min()]).max() / 2.0

mid_x = (X.max() + X.min()) * 0.5
mid_z = (Z.max() + Z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_z - max_range, mid_z + max_range)
ax.axis('off')

plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.show()
