# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sksparse import cholesky
from copy import deepcopy

NUM_OF_LOOP = 10
PLOT_UPDATE_INTERVAL = 0.2 # 500ms

MU    = 0
SIGMA = 0.1

ALPHA_TRANS_VEL   = 0.1
ALPHA_ANGULAR_VEL = 0.1

pose_array = np.empty((0, 2), float)

velocity = np.array([1.0, 1.0], dtype=float)

class node:
    def __init__(self, id, pose):
        self._id   = id
        self._pose = np.array([pose[0], pose[1], pose[2]])

class edge:
    def __init__(self, node_i, node_j, info):
        self.node_i = node(node_i._id, node_i.pose)
        self.node_j = node(node_j._id, node_j.pose)

        self._info_mat = np.identity(3, dtype=float)

    def get_id_pair(self, id_pair):
        id_pair = list(self.node_i._id, self.node_j._id)

    def get_pose_pair(self, pose_pair):
        pose_pair = list(self.node_i._pose, self.node_j._pose)

    def get_info_mat(self, info_mat):
        info_mat = self._info_mat

    def set_pose_pair(self, pose_pair):
        temp_node_i = pose_pair[0]
        temp_node_j = pose_pair[1]

        self.node_i._pose = np.array([temp_node_i[0], temp_node_i[1], temp_node_i[2]])
        self.node_j._pose = np.array([temp_node_i[0], temp_node_i[1], temp_node_j[2]])

    def set_info_mat(self, info_mat):
        self._info_mat[0:3, 0:3] = info_mat

# paran[in]  velocity [vel_trans, vel_angular]
# param[out] cov_mat  covariance of velocity
def calc_cov_mat(velocity, cov_mat):
    cov_mat[0][0] = ALPHA_TRANS_VEL   * (velocity[0] ** 2)
    cov_mat[1][1] = ALPHA_ANGULAR_VEL * (velocity[1] ** 2)

# param[in]  dx           [m]
# param[in]  dtheta       [deg]
# param[out] jacobian_mat 3x6
def calc_jacobian_mat(dx, dtheta, jacobian_mat):
    temp_R  = np.array([[ np.cos(np.radians(dtheta)), -np.sin(np.radians(dtheta))],
                        [ np.sin(np.radians(dtheta)),  np.cos(np.radians(dtheta))]])
    temp_dR = np.array([[-np.sin(np.radians(dtheta)),  np.cos(np.radians(dtheta))],
                        [ np.cos(np.radians(dtheta)), -np.sin(np.radians(dtheta))]])
    jacobian_mat[0:2, 0:2] = -temp_R.T
    jacobian_mat[0:2,   2] =  np.dot(-temp_dR.T, dx)
    jacobian_mat[  2,   2] = -1
    jacobian_mat[0:2, 3:5] =  temp_R.T
    jacobian_mat[  2,   5] =  1

# @param[in]  jacobian_mat
# @param[in]  prev_cov_mat
# @param[out] curr_cov_mat
def calc_curr_cov_mat(jacobian_mat, prev_cov_mat, curr_cov_mat):
    curr_cov_mat = np.dot(np.dot(jacobian_mat, prev_cov_mat), jacobian_mat.T)

""" 
def adjust_pose_graph(node_array, edge_array, hessian_mat):
    hessian_size = range(len(node_array))
    hessian_mat  = np.zeros((hessian_size, hessian_size))
    b = np.zeros((hessian_size, 1))

    prev_cov_mat = np.empty((3, 6), dtype=float)

    # ToDo: pose_arrayでループ回すようにする
    for i in range(hessian_size):
        for j in range(hessian_size):
            # compute Jacobian matrix
            calc_jacobian_mat()
            jacobian_mat_i = jacobian_mat[0:3, 0:3]
            jacobian_mat_j = jacobian_mat[0:3, 3:6]

            calc_curr_cov_mat(jacobian_mat, prev_cov_mat, curr_cov_mat)
            curr_info_mat = np.linalg.inv(curr_cov_mat)

            temp_A = np.dot(jacobian_mat_i.T, prev_cov_mat)
            temp_B = np.dot(jacobian_mat_j.T, prev_cov_mat)           

            # compute Hessian matrix
            hessian_mat[3*i:3*(i+1), 3*i:3*(i+1)] += np.linalg.inv(np.dot(temp_A, jacobian_mat_i))
            hessian_mat[3*i:3*(i+1), 3*j:3*(j+1)] += np.linalg.inv(np.dot(temp_A, jacobian_mat_j))
            hessian_mat[3*j:3*(j+1), 3*i:3*(i+1)] += np.linalg.inv(np.dot(temp_B, jacobian_mat_i))
            hessian_mat[3*j:3*(j+1), 3*j:3*(j+1)] += np.linalg.inv(np.dot(temp_B, jacobian_mat_j))

            calc_error_between_nodes()

            # compute coefficient vector
            b[3*i:3*(i+1), 0] += np.linalg.inv(temp_A, error_i)
            b[3*j:3*(j+1), 0] += np.linalg.inv(temp_A, error_j)

    factor = cholesky(hessian_mat)
    x_hat  = factor(-b)
 """
if __name__ == '__main__':
    jacob = np.zeros((3, 6), dtype=float)
    
    calc_jacobian_mat(np.array([1, 2]), 0, jacob)

    for i in range(NUM_OF_LOOP + 1):
        pose_array = np.append(pose_array, np.array([[np.cos(np.radians(i * 360.0 / NUM_OF_LOOP)), np.sin(np.radians(i * 360.0 / NUM_OF_LOOP))]]), axis=0)

        plt.scatter(pose_array[i][0], pose_array[i][1])
        plt.pause(PLOT_UPDATE_INTERVAL)