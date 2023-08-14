import torch.nn as nn
import pytorch_lightning as pl
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class res_block(nn.Module):
    def __init__(self, num_neurons: int = 1024, use_batchnorm: bool = False, use_dropout: bool = False, dropout=0.5):
        super(res_block, self).__init__()

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.l1 = nn.Linear(num_neurons, num_neurons)
        self.bn1 = nn.LayerNorm(num_neurons)
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(num_neurons, num_neurons)
        self.bn2 = nn.LayerNorm(num_neurons)
        self.d2 = nn.Dropout(dropout)

    def forward(self, x):
        inp = x
        x = self.l1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = nn.LeakyReLU()(x)
        if self.use_dropout:
            x = self.d1(x)
        x = self.l2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = nn.LeakyReLU()(x)
        if self.use_dropout:
            x = self.d2(x)
        x += inp

        return x


class PoseDiscriminator(nn.Module):
    def __init__(self, use_batchnorm=False, num_joints=16, use_dropout=False, dropout=0.5):
        super(PoseDiscriminator, self).__init__()

        self.upscale = nn.Linear(2 * num_joints, 1024)
        self.res_common = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=dropout)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=dropout)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=dropout)
        self.downscale = nn.Linear(1024, 1)

    def forward(self, x):
        x_inp = x

        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))
        #x = nn.LeakyReLU()(self.res_pose1(x))
        x = self.downscale(x)

        return x

class DepthAngleEstimator(nn.Module):
    def __init__(self, use_batchnorm=False, num_joints=16, use_dropout=False, dropout=0.5):
        super(DepthAngleEstimator, self).__init__()

        self.upscale = nn.Linear(2 * num_joints, 1024)
        self.res_common = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=dropout)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=dropout)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=dropout)
        self.res_pose3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=dropout)
        self.res_angle1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=dropout)
        self.res_angle2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=dropout)
        self.res_angle3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=dropout)
        self.downscale = nn.Linear(1024, num_joints)
        self.angles = nn.Linear(1024, 1)
        # self.angles.bias.data[1] = 10.0

    def forward(self, x):
        x_inp = x

        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))

        # pose path
        xd = nn.LeakyReLU()(self.res_pose1(x))
        xd = nn.LeakyReLU()(self.res_pose2(xd))
        xd = nn.LeakyReLU()(self.res_pose3(xd))
        xd = self.downscale(xd)
        # xd = self.tanh(xd)

        # depth path
        xa = nn.LeakyReLU()(self.res_angle1(x))
        xa = nn.LeakyReLU()(self.res_angle2(xa))
        xa = nn.LeakyReLU()(self.res_angle3(xa))
        xa = self.angles(xa)

        return xd, xa



class Leg_Lifter(nn.Module):
    def __init__(self, use_batchnorm=False, num_joints=7, use_dropout=False, d_rate=0.5):
        super(Leg_Lifter, self).__init__()
        self.upscale = nn.Linear(2 * num_joints, 1024)
        self.res_common = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=use_dropout)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=use_dropout)
        self.res_pose3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=use_dropout)
        self.res_angle1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.res_angle2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.res_angle3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.downscale = nn.Linear(1024, num_joints)
        self.angles = nn.Linear(1024, 1)
        # self.angles.bias.data[1] = 10.0

    def forward(self, x):
        x_inp = x

        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))

        # pose path
        xd = nn.LeakyReLU()(self.res_pose1(x))
        xd = nn.LeakyReLU()(self.res_pose2(xd))
        xd = nn.LeakyReLU()(self.res_pose3(xd))
        xd = self.downscale(xd)
        # xd = self.tanh(xd)

        # depth path
        xa = nn.LeakyReLU()(self.res_angle1(x))
        xa = nn.LeakyReLU()(self.res_angle2(xa))
        xa = nn.LeakyReLU()(self.res_angle3(xa))
        xa = self.angles(xa)

        return xd, xa


class Torso_Lifter(nn.Module):
    def __init__(self, use_batchnorm=False, num_joints=10, use_dropout=False, d_rate=0.5):
        super(Torso_Lifter, self).__init__()
        self.upscale = nn.Linear(2 * num_joints, 1024)
        self.res_common = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=use_dropout)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=use_dropout)
        self.res_pose3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=use_dropout)
        self.res_angle1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.res_angle2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.res_angle3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.downscale = nn.Linear(1024, num_joints)
        self.angles = nn.Linear(1024, 1)
        # self.angles.bias.data[1] = 10.0

    def forward(self, x):
        x_inp = x

        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))

        # pose path
        xd = nn.LeakyReLU()(self.res_pose1(x))
        xd = nn.LeakyReLU()(self.res_pose2(xd))
        xd = nn.LeakyReLU()(self.res_pose3(xd))
        xd = self.downscale(xd)
        # xd = self.tanh(xd)

        # depth path
        xa = nn.LeakyReLU()(self.res_angle1(x))
        xa = nn.LeakyReLU()(self.res_angle2(xa))
        xa = nn.LeakyReLU()(self.res_angle3(xa))
        xa = self.angles(xa)

        return xd, xa

class Left_Right_Lifter(nn.Module):
    def __init__(self, use_batchnorm=False, num_joints=11, use_dropout=False, d_rate=0.5):
        super(Left_Right_Lifter, self).__init__()
        self.upscale = nn.Linear(2 * num_joints, 1024)
        self.res_common = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=use_dropout)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=use_dropout)
        self.res_pose3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                   dropout=use_dropout)
        self.res_angle1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.res_angle2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.res_angle3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024, use_dropout=use_dropout,
                                    dropout=use_dropout)
        self.downscale = nn.Linear(1024, num_joints)
        self.angles = nn.Linear(1024, 1)
        # self.angles.bias.data[1] = 10.0

    def forward(self, x):
        x_inp = x

        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))

        # pose path
        xd = nn.LeakyReLU()(self.res_pose1(x))
        xd = nn.LeakyReLU()(self.res_pose2(xd))
        xd = nn.LeakyReLU()(self.res_pose3(xd))
        xd = self.downscale(xd)
        # xd = self.tanh(xd)

        # depth path
        xa = nn.LeakyReLU()(self.res_angle1(x))
        xa = nn.LeakyReLU()(self.res_angle2(xa))
        xa = nn.LeakyReLU()(self.res_angle3(xa))
        xa = self.angles(xa)

        return xd, xa



class Occluded_Limb_Predictor(nn.Module):
    def __init__(self, use_batchnorm=False, num_joints=10):
        super(Occluded_Limb_Predictor, self).__init__()
        self.upscale = nn.Linear(3 * num_joints, 1024)
        self.res_common = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.downscale = nn.Linear(1024, 3 * 3)

    def forward(self, x):
        x_inp = x

        # pose path
        x = self.upscale(x)
        xd = nn.LeakyReLU()(self.res_pose1(x))
        xd = nn.LeakyReLU()(self.res_pose2(xd))
        xd = nn.LeakyReLU()(self.res_pose3(xd))
        xd = self.downscale(xd)

        return xd


class Occluded_Legs_Predictor(nn.Module):
    def __init__(self, use_batchnorm=False, num_joints=10):
        super(Occluded_Legs_Predictor, self).__init__()
        self.upscale = nn.Linear(3 * num_joints, 1024)
        self.res_common = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.downscale = nn.Linear(1024, 3 * 6)

    def forward(self, x):
        x_inp = x

        # pose path
        x = self.upscale(x)
        xd = nn.LeakyReLU()(self.res_pose1(x))
        xd = nn.LeakyReLU()(self.res_pose2(xd))
        xd = nn.LeakyReLU()(self.res_pose3(xd))
        xd = self.downscale(xd)

        return xd


class Occluded_Torso_Predictor(nn.Module):
    def __init__(self, use_batchnorm=False, num_joints=10):
        super(Occluded_Torso_Predictor, self).__init__()
        self.upscale = nn.Linear(3 * num_joints, 1024)
        self.res_common = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.downscale = nn.Linear(1024, 3 * 10)

    def forward(self, x):
        x = self.upscale(x)
        xd = nn.LeakyReLU()(self.res_pose1(x))
        xd = nn.LeakyReLU()(self.res_pose2(xd))
        xd = nn.LeakyReLU()(self.res_pose3(xd))
        xd = self.downscale(xd)

        return xd


class Occluded_Left_Right_Predictor(nn.Module):
    def __init__(self, use_batchnorm=False, num_joints=10):
        super(Occluded_Left_Right_Predictor, self).__init__()
        self.upscale = nn.Linear(3 * num_joints, 1024)
        self.res_common = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.res_pose3 = res_block(use_batchnorm=use_batchnorm, num_neurons=1024)
        self.downscale = nn.Linear(1024, 3 * 6)

    def forward(self, x):
        # pose path
        x = self.upscale(x)
        xd = nn.LeakyReLU()(self.res_pose1(x))
        xd = nn.LeakyReLU()(self.res_pose2(xd))
        xd = nn.LeakyReLU()(self.res_pose3(xd))
        xd = self.downscale(xd)

        return xd
