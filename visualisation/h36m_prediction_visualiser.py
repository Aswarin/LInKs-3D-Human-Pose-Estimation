import torch.nn
import torch.optim
from lift_and_fill_models.utils.helpers import *
from lift_and_fill_models.utils.models_def import Attention_Leg_Lifter, Attention_Left_Right_Lifter, Attention_Torso_Lifter, Occluded_Limb_Predictor, Occluded_Torso_Predictor, Occluded_Legs_Predictor
import matplotlib as mpl
import matplotlib.pyplot as plt
torch.manual_seed(19)
import torch.optim
import torch.nn
import torch.optim
from sklearn.decomposition import PCA
from lift_and_fill_models.utils.helpers import *
from torch.utils.data import Dataset
import pickle
from matplotlib import animation

#173913 -80 - 1609
#175522
pose_choice = 320

def combine_pose_and_limb(pose, limb, which_limb):
    limb = limb.reshape(-1, 3, 3)
    pose = pose.reshape(-1, 3, 14)
    if which_limb == 'll':
        full_pose = torch.cat((pose[:, :, :4], limb, pose[:, :, 4:]), dim=2)
    elif which_limb == 'rl':
        full_pose = torch.cat((pose[:, :, :1], limb, pose[:, :, 1:]), dim=2)
    elif which_limb == 'la':
        full_pose = torch.cat((pose[:, :, :11], limb, pose[:, :, 11:]), dim=2)
    elif which_limb == 'ra':
        full_pose = torch.cat((pose, limb), dim=2)
    return full_pose.reshape(-1, 51)

# trained_leg_lifting_network = Attention_Leg_Lifter(use_batchnorm=False, num_joints=7).cuda()
# trained_torso_lifting_network = Attention_Torso_Lifter(use_batchnorm=False, num_joints=10).cuda()
trained_left_lifting_network = Attention_Left_Right_Lifter(use_batchnorm=False, num_joints=11, use_dropout=False,
                                                           d_rate=0.25, num_heads=2).cuda()
trained_right_lifting_network = Attention_Left_Right_Lifter(use_batchnorm=False, num_joints=11, use_dropout=False,
                                                            d_rate=0.25, num_heads=2).cuda()
left_leg_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=14).cuda()
right_leg_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=14).cuda()
left_arm_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=14).cuda()
right_arm_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=14).cuda()
torso_predictor = Occluded_Torso_Predictor(use_batchnorm=False, num_joints=7).cuda()
legs_predictor = Occluded_Legs_Predictor(use_batchnorm=False, num_joints=11).cuda()

# trained_leg_lifting_network.load_state_dict(
#     torch.load('../lift_and_fill_models/best_weights_lifting_model/attention_leg_lifter_new_norm_flow_with_sampling33.9.pt'))
# trained_torso_lifting_network.load_state_dict(
#     torch.load('../lift_and_fill_models/best_weights_lifting_model/attention_torso_lifter_new_norm_flow_with_sampling33.9.pt'))
trained_left_lifting_network.load_state_dict(
    torch.load('../models/mixed_left_lifter_final.pt'), strict=False)

trained_right_lifting_network.load_state_dict(
    torch.load('../models/mixed_right_lifter_final.pt'), strict=False)

# trained_right_lifting_network.load_state_dict(
#     torch.load('../lift_and_fill_models/best_weights_lifting_model/right_side_33.1_with_attention_and_sampling.pt'), strict=False)
# trained_left_lifting_network.load_state_dict(
#     torch.load('../lift_and_fill_models/best_weights_lifting_model/left_side_33.1_with_sampling_and_attention.pt'), strict=False)



left_leg_predictor.load_state_dict(torch.load('../models/occlusion_model_weights/left_leg_estimator.pt'))
right_leg_predictor.load_state_dict(torch.load('../models/occlusion_model_weights/right_leg_estimator.pt'))
left_arm_predictor.load_state_dict(torch.load('../models/occlusion_model_weights/left_arm_estimator.pt'))
right_arm_predictor.load_state_dict(torch.load('../models/occlusion_model_weights/right_arm_estimator.pt'))
torso_predictor.load_state_dict(torch.load('../models/occlusion_model_weights/torso_estimator.pt'))
legs_predictor.load_state_dict(torch.load('../models/occlusion_model_weights/both_legs_estimator.pt'))
trained_right_lifting_network.eval()
trained_left_lifting_network.eval()
left_arm_predictor.eval()
right_arm_predictor.eval()
left_leg_predictor.eval()
right_leg_predictor.eval()
legs_predictor.eval()
torso_predictor.eval()


def pmpjpe(p_ref, p, reflection=False):
    # reshape pose if necessary
    if p.shape[0] == 1:
        p = p.reshape(3, int(p.shape[1] / 3))

    if p_ref.shape[0] == 1:
        p_ref = p_ref.reshape(3, int(p_ref.shape[1] / 3))

    d, Z, tform = procrustes(p_ref.T, p.T, reflection=reflection)
    return Z

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform

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
            three_d_joints.append(data[s]['poses_3d_univ'])
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

datafile = '../../EVAL_DATA/h36m_interesting_frames_mixed_order.pkl'

# train_data = H36M_Data(datafile, train=True, get_pca=True, normalize_func=normalize_head, get_2dgt=True,
#                                  subjects=['S1', 'S5', 'S7', 'S6', 'S8'])
# pca = train_data.pca
test_data = H36M_Data(datafile, train=False, normalize_func=normalize_head_test, get_2dgt=True,
                                subjects=['S9'])


h36m_data_2d = torch.tensor(test_data.data['poses_2d'][:]).cuda()
h36m_data_3d = np.array(test_data.data['poses_3d'][:])

legs = h36m_data_2d.reshape(-1, 2, 17)[:, :, :7].reshape(-1, 14)
torso = h36m_data_2d.reshape(-1, 2, 17)[:, :, 7:].reshape(-1, 20)

left_mpi_2d, right_mpi_2d = split_data_left_right_v2(h36m_data_2d)

left_pred_z, _ = trained_left_lifting_network(left_mpi_2d[pose_choice].reshape(-1, 22))
right_pred_z, _ = trained_right_lifting_network(right_mpi_2d[pose_choice].reshape(-1, 22))
# leg_pred_z, _ = trained_leg_lifting_network(legs)
# torso_pred_z, _ = trained_torso_lifting_network(torso)

"""Decide which model you want by uncommenting the below"""
#pred_test_z = torch.cat((leg_pred_z, torso_pred_z), dim=1)
pred_test_z = combine_left_right_pred_1d(left_pred_z, right_pred_z, choice='left').reshape(-1, 17)

pred_test_z[:, 0] = 0.0

pred_test_depth = pred_test_z + 10

pred_test_poses = torch.cat(
    ((h36m_data_2d[pose_choice].reshape(-1, 2, 17) * pred_test_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34),
     pred_test_depth), dim=1)

pred_test_poses = pred_test_poses.reshape(-1, 3, 17) - pred_test_poses.reshape(-1, 3, 17)[:, :, [0]]

"""Decide if you want occlusion here"""
#
input_3d_no_left_arm = torch.cat((pred_test_poses[:, :, :11], pred_test_poses[:, :, 14:]), dim=2).reshape(-1, 42)
input_3d_no_right_arm = pred_test_poses[:, :, :14].reshape(-1, 42)
input_3d_no_left_leg = torch.cat((pred_test_poses[:, :, :4], pred_test_poses[:, :, 7:]), dim=2).reshape(-1, 42)
input_3d_no_right_leg = torch.cat((pred_test_poses[:, :, :1], pred_test_poses[:, :, 4:]), dim=2).reshape(-1, 42)
input_3d_no_torso = pred_test_poses[:, :, :7].reshape(-1, 21)
input_3d_no_legs = torch.cat((pred_test_poses[:, :, :1], pred_test_poses[:, :, 7:]), dim=2).reshape(-1, 33)

left_leg = left_leg_predictor(input_3d_no_left_leg)
right_leg = right_leg_predictor(input_3d_no_right_leg)
left_arm = left_arm_predictor(input_3d_no_right_arm)
right_arm = right_arm_predictor(input_3d_no_left_arm)
both_legs = legs_predictor(input_3d_no_legs)
torso = torso_predictor(input_3d_no_torso)
full_pose_la = combine_pose_and_limb(input_3d_no_left_arm, left_arm, 'la')
full_pose_ra = combine_pose_and_limb(input_3d_no_right_arm, right_arm, 'ra')
full_pose_ll = combine_pose_and_limb(input_3d_no_left_leg, left_leg, 'll')
full_pose_rl = combine_pose_and_limb(input_3d_no_right_leg, right_leg, 'rl')
full_pose_legs = torch.cat((input_3d_no_legs.reshape(-1, 3, 11)[:, :, :1], both_legs.reshape(-1, 3, 6), input_3d_no_legs.reshape(-1, 3, 11)[:, :, 1:]), dim=2).reshape(-1, 51)
full_pose_torso = torch.cat((input_3d_no_torso.reshape(-1, 3, 7), torso.reshape(-1, 3, 10)), dim=2).reshape(-1, 51)
pred_test_poses = full_pose_torso
"""end occlusion"""

pred_test_poses = pred_test_poses.detach().cpu().numpy()

h36m_data_3d = h36m_data_3d.reshape(-1, 3, 17) - h36m_data_3d.reshape(-1, 3, 17)[:, :, [0]]

transformed_pose = pmpjpe(h36m_data_3d[pose_choice].reshape(1, 51), pred_test_poses.reshape(1, 51), reflection='best')

pose = transformed_pose
gt_pose = np.array(h36m_data_3d[pose_choice]).reshape(3, 17).transpose(1, 0)
check = pose
buff_large = np.zeros((32, 3))
buff_large_gt = np.zeros((32, 3))
buff_large[(0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27), :] = pose
buff_large_gt[(0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27), :] = gt_pose
pose = buff_large.transpose()
gt_pose = buff_large_gt.transpose()
#
kin = np.array([[0, 12], [12, 13], [13, 14], [15, 14], [13, 17], [17, 18], [18, 19], [13, 25], [25, 26], [26, 27], [0, 1], [1, 2], [2, 3], [0, 6], [6, 7], [7, 8]])
# kin = np.array([[0, 12], [12, 13], [13, 14], [15, 14], [13, 17], [17, 18], [18, 19],
#                 [0, 1], [1, 2], [2, 3], [0, 6], [6, 7], [7, 8]])
order = np.array([0, 2, 1])

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.view_init(azim=-45, elev=15)

for link in kin:
    ax.plot(pose[0, link], pose[2, link], -pose[1, link], linewidth=5.0)
    #ax.plot(gt_pose[0, link], gt_pose[2, link], -gt_pose[1, link], linewidth=5.0, color='green')

ax.legend()
# ax.set_xlabel('X')
# ax.set_ylabel('Z')
# ax.set_zlabel('Y')
ax.set_aspect('auto')
#ax.axis('off')

X = pose[0, :]
Y = pose[2, :]
Z = -pose[1, :]
Xt = gt_pose[0, :]
Yt = gt_pose[2, :]
Zt = -gt_pose[1, :]
max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()/2.0

mid_x = (X.max() + X.min()) * 0.5
mid_y = (Y.max() + Y.min()) * 0.5
mid_z = (Z.max() + Z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.view_init(elev=19, azim=-59)
ax.view_init(elev=19, azim=(180-59))

#plt.title('Predicted 3D Pose Estimate', y=0.95, fontsize=16)

plt.tick_params(left = False, right = False , labelleft = False,
                labelbottom = False, bottom = False)

plt.show()

# def init():
#     ax.scatter(X, Y, Z, marker='o', s=20, c="blue", alpha=0.6)
#     ax.scatter(Xt, Yt, Zt, marker='o', s=20, c="red", alpha=0.6)
#     return fig,
#
# def animate(i):
#     ax.view_init(elev=10., azim=i)
#     return fig,
#
# def animate(i):
#     ax.view_init(elev=10., azim=i)
#     return fig,
#
# # Animate
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=360, interval=20, blit=True)
# # Save
# anim.save('ground_truth.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

