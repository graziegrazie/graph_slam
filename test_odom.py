# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

LOOP_INTERVAL = 0.2 # 200ms
MU    = 0
SIGMA = 1

X = 0
Y = 1

if __name__ == '__main__':
    init_pose = np.array([0, 0], dtype=float)
    prev_pose = init_pose
    curr_pose = np.array([0.0, 0.0], dtype=float)

    V = np.array([1, 2])

    # 初期値をプロット
    plt.scatter(curr_pose[0], curr_pose[1], marker="*", s=100)

    for i in range(10):
        noise = np.random.normal(MU, SIGMA, (1, 2))[0]

        delta_trans = np.array([V[X] + noise[X], V[Y] + noise[Y]])

        curr_pose[0] = prev_pose[0] + delta_trans[0]
        curr_pose[1] = prev_pose[1] + delta_trans[1]

        plt.scatter(curr_pose[0], curr_pose[1], marker="*", s=100)
        plt.quiver(prev_pose[0],  prev_pose[1], delta_trans[0], delta_trans[1], angles="xy", scale_units="xy", scale=1, color="gray",label="guess robot motion")

        prev_pose  = np.array([curr_pose[0], curr_pose[1]], dtype=float)

        plt.pause(LOOP_INTERVAL)