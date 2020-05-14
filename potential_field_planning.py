"""
@Author: Malintha Fernando
Multi robot path planning based on Artificial Potential Fields
Using Gaussian Potentials
"""

import numpy as np
import math
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import glob

OBSLEN = 5
AREA_WIDTH = 30.0  #
show_animation = True


def set_obstacle_potential(xw, yw, ox, oy, res):
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]
    leni = int(round(OBSLEN / res))
    sidelen = int(round(leni / 2))
    for o in range(len(ox)):
        xval = int(ox[o] / res)
        yval = int(oy[o] / res)
        for l in range(-sidelen, sidelen):
            for w in range(-sidelen, sidelen):
                pmap[w + yval][l + xval] = 10

    return gaussian_filter(pmap, sigma=1)


def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def calc_goal_potential(pos, goal):
    return 3 * math.exp(la.norm(np.array(pos) - np.array(goal)) / 30)


def calc_static_potential_field(ox, oy, gx, gy, res, rr):
    minx = 0
    miny = 0
    maxx = AREA_WIDTH
    maxy = AREA_WIDTH
    xw = int(round((maxx - minx) / res))
    yw = int(round((maxy - miny) / res))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        x = ix * res + minx
        for iy in range(yw):
            y = iy * res + miny
            pmap[iy][ix] = calc_goal_potential([x, y], [gx, gy])

    pmap = np.array(pmap)
    op = set_obstacle_potential(xw, yw, ox, oy, res)

    return pmap + op, minx, miny, maxx, maxy


def get_interaction_potential(posA, posB):
    a = 0.2
    b = 2
    ca = 8
    cr = 1
    dis = la.norm(np.array(posA) - np.array(posB))
    att_pot = -a * math.exp(-dis / ca)
    rep_pot = b * math.exp(-dis / cr)
    return att_pot + rep_pot


def cal_dynamic_potential_field(curr_pos, id, xw, yw, res):
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]
    for i in range(yw):
        for j in range(xw):
            for r in range(len(curr_pos)):
                if r == id:
                    continue
                pmap[j][i] += 3 * get_interaction_potential([i, j], (np.array(curr_pos[r]) / res)) + 2
    return pmap


def calc_next_position(id, curr_pos, res, pmap, xvals, yvals, it):
    minx = xvals[0]
    maxx = xvals[1]
    miny = yvals[0]
    maxy = yvals[1]

    xw = int(round((maxx - minx) / res))
    yw = int(round((maxy - miny) / res))

    pdmap = cal_dynamic_potential_field(curr_pos, id, xw, yw, res)
    synthesized_map = np.array(pmap) + np.array(pdmap)

    ix = round((curr_pos[id][0] - minx) / res)
    iy = round((curr_pos[id][1] - miny) / res)
    if show_animation:
        if id == 0:
            colstr = "*r"
        elif id == 1:
            colstr = "*g"
        else:
            colstr = "*b"
        plt.plot(ix, iy, colstr)

    motion = get_motion_model()
    minp = float("inf")
    minix, miniy = -1, -1
    for i, _ in enumerate(motion):
        inx = int(ix + motion[i][0])
        iny = int(iy + motion[i][1])
        if inx >= len(pmap) or iny >= len(pmap[0]):
            p = float("inf")  # outside area
        else:
            p = synthesized_map[iny][inx]
        if minp > p:
            minp = p
            minix = inx
            miniy = iny
    ix = minix
    iy = miniy
    xp = ix * res + minx
    yp = iy * res + miny
    curr_pos[id] = [xp, yp]

    if show_animation:
        plt.pause(0.0001)
        if id == len(curr_pos)-1:
            plt.savefig('data/im_'+str(it))
    return curr_pos


def draw_heatmap(data):
    data = np.array(data)
    plt.pcolor(data, vmax=20.0, cmap=plt.cm.Blues)


def main():
    print("potential_field_planning start")

    curr_pos = [[5, 8],
                [3, 5],
                [8, 6]]

    gx = 25.0  # goal x position [m]
    gy = 25.0  # goal y position [m]

    grid_size = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]

    ox = [15]  # obstacle x position list [m]
    oy = [15]  # obstacle y position list [m]

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    pmap, minx, miny, maxx, maxy = calc_static_potential_field(ox, oy, gx, gy, grid_size, robot_radius)

    gix = round((gx - minx) / grid_size)
    giy = round((gy - miny) / grid_size)

    if show_animation:
        draw_heatmap(pmap)
        plt.plot(gix, giy, "-ok")
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
    xvals = [minx, maxx]
    yvals = [miny, maxy]

    # num of iterations
    for it in range(60):
        for r in range(len(curr_pos)):
            curr_pos = calc_next_position(r, curr_pos, grid_size, pmap, xvals, yvals, it)


if __name__ == '__main__':
    files = glob.glob('./data/*')
    for f in files:
        os.remove(f)
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")