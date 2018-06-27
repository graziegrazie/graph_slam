# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from scikits.sparse.cholmod import cholesky

NUM_OF_LOOP = 10
PLOT_UPDATE_INTERVAL = 0.2 # 500ms

MU    = 0
SIGMA = 0.1
ALPHA_TRANS_VEL   = 0.1
ALPHA_ANGULAR_VEL = 0.1

pose_array = np.empty((0, 2), float)

velocity = np.array([1.0, 1.0], dtype=float)

class PoseGraph:
    def __init__(self):
        self._origin_node  = Node(0, Pose2D()) # define origin at (x, y, th) = [0, 0, 0]
        self._nodes        = list(self._origin_node)
        self._edges        = list()
        self._num_of_nodes = len(self._nodes)

        self._prev_cov_odometry = np.zeros((3, 3), dtype=float)

    def storePoseAsNode(self, x, y, th):
        latest_node = Node(self._num_of_nodes - 1, Pose2D(x, y, th)) # Be careful! Node ID and number of lists are different!
        self._nodes.append(latest_node)

        self._num_of_nodes = self._num_of_nodes + 1

    def storeEdge(self, prev_node, post_node, movement, info):
        latest_edge = Edge(prev_node, post_node, movement, info)
        self._edges.append(latest_edge)

    # cov_fuse = 1 / ((1 / cov_measure) + (1 / cov_odometry)) = cov_fuse = 1 / (info_measure + info_odometry)
    def calc_covariance(self, cov_measurement, cov_odometry, cov_fusion):
        cov_fuse = np.linalg.inv(cov_measurement) + np.linalg.inv(cov_odometry)
        cov_fuse = np.linalg.inv(cov_fuse)

    # @param[in]  current_scan_point   : scan robot got just now
    # @param[in]  reference_scan_point : reference scan for matching
    # @param[in]  sensor_pose          : coordinate of sensor
    # @param[out] vertical_distance    : vertical distance between current_scan and reference_scan
    def calc_vertical_distance(self, sensor_pose, current_scan_point, reference_scan_point, vertical_distance):
        tx = sensor_pose[0]
        ty = sensor_pose[1]
        th = sensor_pose[2]

        x = np.cos(th) * current_scan_point.x - np.sin(th) * current_scan_point.y + tx # clpを推定位置で座標変換
        y = np.sin(th) * current_scan_point.x + np.cos(th) * current_scan_point.y + ty

        vertical_distance = (x - reference_scan_point.x) * reference_scan_point.nx + (y - reference_scan_point.y) * reference_scan_point.ny # 座標変換した点からrlpへの垂直距離

    # 垂直距離を用いた観測モデルの式
    def calc_covariance_measurement(self, sensor_pose, current_scan, reference_scan, cov_measurement):
        ISOLATE = 3

        # INFO: These variables are parameter for this function
        dd      = 0.00001
        dth     = 0.00001

        temp_sensor_pose = Pose2D(sensor_pose.x, sensor_pose.y, sensor_pose.theta)
        
        vertical_distance_standard = np.zeros((3, 3), dtype=float)
        vertical_distance_dx       = np.zeros((3, 3), dtype=float)
        vertical_distance_dy       = np.zeros((3, 3), dtype=float)
        vertical_distance_dth      = np.zeros((3, 3), dtype=float)

        cov_measurement = np.zeros((3, 3), dtype=float)

        for current_scan_point, reference_scan_point in zip(current_scan, reference_scan):
            # TODO: this is specific for LittelSLAM data format
            if reference_scan_point.type == ISOLATE:
                continue
            else:
                pass
            
            calc_vertical_distance(temp_sensor_pose, current_scan_point, reference_scan_point, vertical_distance_standard)

            temp_sensor_pose.x = temp_sensor_pose.x + dd
            calc_vertical_distance(temp_sensor_pose, current_scan_point, reference_scan_point, vertical_distance_dx)

            temp_sensor_pose.x = temp_sensor_pose.x - dd
            temp_sensor_pose.y = temp_sensor_pose.y + dd
            calc_vertical_distance(temp_sensor_pose, current_scan_point, reference_scan_point, vertical_distance_dy)

            temp_sensor_pose.y     = temp_sensor_pose.y - dd
            temp_sensor_pose.theta = temp_sensor_pose.y + dth
            calc_vertical_distance(temp_sensor_pose, current_scan_point, reference_scan_point, vertical_distance_dth)

            Jx  = (vertical_distance_dx  - vertical_distance_standard) / dd
            Jy  = (vertical_distance_dy  - vertical_distance_standard) / dd
            Jth = (vertical_distance_dth - vertical_distance_standard) / dth

            cov_measurement[0][0] += np.dot(Jx  * Jx)
            cov_measurement[0][1] += np.dot(Jx  * Jy)
            cov_measurement[0][2] += np.dot(Jx  * Jth)
            cov_measurement[1][1] += np.dot(Jy  * Jy)
            cov_measurement[1][2] += np.dot(Jy  * Jth)
            cov_measurement[2][2] += np.dot(Jth * Jth)

        # Utilize that Hessian is Symmetric Matrix
        cov_measurement[1][0] = cov_measurement[0][1]
        cov_measurement[2][0] = cov_measurement[0][2]
        cov_measurement[2][1] = cov_measurement[1][2]

    def calc_covariance_odometry(self, pose, velocity, dt, cov_odometry):
        # These value is Const. values
        ALPHA1 = 1
        ALPHA2 = 1.2
        ALPHA3 = 0.8
        ALPHA4 = 5

        theta  = pose._theta

        v = velocity[0] # translation velocity
        w = velocity[1] # angular velocity

        Jx = np.array([[1, 0, -v * dt * np.sin(theta)],
                       [0, 1,  v * dt * np.cos(theta)],
                       [0, 0,  1                     ]])
        Ju = np.array([[dt * np.cos(theta), 0 ],
                       [dt * np.sin(theta), 0 ],
                       [0,                  dt]])

        # TODO: Probabilistic Roboticsの速度制御モデルの共分散の式を見る
        Sigma_u = np.array([[ALPHA1 * (v ** 2) + ALPHA2 * (w ** 2), 0                                    ],
                            [0,                                     ALPHA3 * (v ** 2) + ALPHA4 * (w ** 2)]])

        cov_odometry = np.dot(np.dot(Jx, self._prev_cov_odometry), Jx.T) + np.dot(np.dot(Ju, Sigma_u), Ju.T)
        self._prev_cov_odometry[:] = cov_odometry

class Pose2D:
    def __init__(self, x=0, y=0, theta=0):
        self.x     = x
        self.y     = y
        self.theta = theta

class Node:
    def __init__(self, id, pose):
        self.id   = id
        self.pose = Pose2D(pose.x, pose.y, pose.theta)

class Edge:
    def __init__(self, prev_node, post_node, movement, info):
        self.prev_node = Node(prev_node.id, prev_node.pose)
        self.post_node = Node(post_node.id, post_node.pose)
        self.movement  = Pose2D(movement.x, movement.y, movement.theta) # movement from Pose2D

        self.info = np.empty((3, 3), dtype=float)
        self.info[:] = info

def adjust_pose_graph(x_hat, constraints):
    num_of_nodes = len(x_hat)
    jacobian = np.empty((3, 6), dtype=float)

    F = 0
    threshold = 0.1

    while not (F < threshold):
        hessian     = np.empty((num_of_nodes * 3, num_of_nodes * 3), dtype=float)
        info_vector = np.empty((num_of_nodes * 3), dtype=float)

        F = 0

        for c in constraints:
            i = c[0]._prev_node._id
            j = c[0]._post_node._id

            node_i = x_hat[i]
            node_j = x_hat[j]

            # Following section is programmed with motion model
            calc_jacobian(c, jacobian)

            A_ij     = jacobian[0:3, 0:3]
            B_ij     = jacobian[0:3, 3:6]
            Sigma_ij = c[1]

            # update H
            hessian[i*3:(i+1)*3, i*3:(i+1)*3] += np.dot(np.dot(A_ij.T, Sigma_ij), A_ij)
            hessian[i*3:(i+1)*3, j*3:(j+1)*3] += np.dot(np.dot(A_ij.T, Sigma_ij), B_ij)
            hessian[j*3:(j+1)*3, i*3:(i+1)*3] += np.dot(np.dot(B_ij.T, Sigma_ij), A_ij)
            hessian[j*3:(j+1)*3, j*3:(j+1)*3] += np.dot(np.dot(B_ij.T, Sigma_ij), B_ij)
            
            # update b
            info_vector[i*3:(i+1)*3] += np.dot(np.dot(A_ij.T, Sigma_ij), A_ij)
            info_vector[j*3:(j+1)*3] += np.dot(np.dot(A_ij.T, Sigma_ij), A_ij)
        
        hessian[0][0] += 10000 # to keep initial coordinates

        factor  = cholesky(hessian)
        delta_x = factor(info_vector)
        x_hat  += delta_x

        # calculate F(x) = e.T * Omega * e
        error = (node_i.pose - node_j.pose) - c[0].edge.pose
        F += np.dot(np.dot(error.T, c[1]), error)

def calc_jacobian(constaint, jacobian):
    global velocity

    theta1 = constaint[0][2]
    cos1   = np.cos(theta1)
    sin1   = np.sin(theta1)

    u = velocity[0]
    v = velocity[1]

    # This equation is from "SLAM Introduction" by Tomono
    jacobian = np.array([[1, 0, -(u * sin1 + v * cos1), cos1, -sin1, 0],
                         [0, 1,  (u * cos1 - v * sin1), sin1,  cos1, 0],
                         [0, 0,  1,                     0,     0,    1]])

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

# ToDo: You should check this implementation is correct or not.
# @param[in]  jacobian_mat
# @param[in]  prev_cov_mat
# @param[out] curr_cov_mat
def calc_curr_cov_mat(jacobian_mat, prev_cov_mat, velocity, curr_cov_mat):
    Jx_t = jacobian_mat[0:3, 0:3]
    Ju_t = jacobian_mat[0:3, 3:6]

    cov_u = np.zeros((2, 2))
    calc_cov_mat(velocity, cov_u)

    curr_cov_mat = np.dot(np.dot(Jx_t, prev_cov_mat), Jx_t.T) + np.dot(np.dot(Ju_t, cov_u), Ju_t.T)

    hessian_size = range(len(node_array))
    hessian_mat  = np.zeros((hessian_size, hessian_size))

    b = np.zeros((hessian_size, 1))

    prev_cov_mat = np.empty((3, 6), dtype=float)
    curr_cov_mat = np.empty((3, 6), dtype=float)

    jacobian_mat = np.empty((3, 6), dtype=float)

    # ToDo: pose_arrayでループ回すようにする
    for i in range(hessian_size):
        for j in range(hessian_size):
            dx = 0
            dtheta = 0
            
            calc_jacobian_mat(dx, dtheta, jacobian_mat)
            jacobian_mat_i = jacobian_mat[0:3, 0:3]
            jacobian_mat_j = jacobian_mat[0:3, 3:6]

            calc_curr_cov_mat(jacobian_mat, prev_cov_mat, velocity, curr_cov_mat)
            curr_info_mat = np.linalg.inv(curr_cov_mat)

            temp_A = np.dot(jacobian_mat_i.T, prev_cov_mat)
            temp_B = np.dot(jacobian_mat_j.T, prev_cov_mat)           

            # compute Hessian matrix
            hessian_mat[3*i:3*(i+1), 3*i:3*(i+1)] += np.linalg.inv(np.dot(temp_A, jacobian_mat_i))
            hessian_mat[3*i:3*(i+1), 3*j:3*(j+1)] += np.linalg.inv(np.dot(temp_A, jacobian_mat_j))
            hessian_mat[3*j:3*(j+1), 3*i:3*(i+1)] += np.linalg.inv(np.dot(temp_B, jacobian_mat_i))
            hessian_mat[3*j:3*(j+1), 3*j:3*(j+1)] += np.linalg.inv(np.dot(temp_B, jacobian_mat_j))

            # まず先にノードとエラーのセットを作っておく必要がある
            # calc_error_between_nodes()

            # compute coefficient vector
            b[3*i:3*(i+1), 0] += np.linalg.inv(temp_A, error_ij)
            b[3*j:3*(j+1), 0] += np.linalg.inv(temp_B, error_ij)

    factor = np.linalg.cholesky(hessian_mat)
    x_hat  = factor(-b)
 
if __name__ == '__main__':
    jacob = np.zeros((3, 6), dtype=float)
    
    calc_jacobian_mat(np.array([1, 2]), 0, jacob)

    for i in range(NUM_OF_LOOP + 1):
        pose_array = np.append(pose_array, np.array([[np.cos(np.radians(i * 360.0 / NUM_OF_LOOP)), np.sin(np.radians(i * 360.0 / NUM_OF_LOOP))]]), axis=0)

        plt.scatter(pose_array[i][0], pose_array[i][1])
        plt.pause(PLOT_UPDATE_INTERVAL)