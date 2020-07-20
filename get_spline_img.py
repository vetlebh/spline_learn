import tensorflow as tf
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy.integrate import simps
from scipy.interpolate import BSpline
from cv2 import floodFill


def calc_intersect(point):
    #cell_height = point[1]-int(point[1])
    #return cell_height*255.0
    return 255.0


def calc_stepsize(nd, d, p_i, delta_t):
    """
    Based on theroretical estimate, see report
    """
    return 1.0/((nd+d)*np.max(np.sum(p_i, axis=0)))
    #return delta_t/(np.sum(np.sqrt(np.sum(np.square(p_i), axis=1))))



def floodFill_start(img, row):
    """
    Should be cleaned
    """
    """
    Iterates outwards from start row to find point satisfied by
    ray-casting algortihm to be inside spline
    """
    if (row >= 256 or row < 0):
        return False
    next_row = int(row + 10)
    intersect = np.where(img[row, :] > 0.0)[0]
    nr_intersect = intersect.shape[0]
    if (nr_intersect > 0):
        if (intersect[0] > 0):
            return (0, row)
        else:
            for i in range (0, nr_intersect-2, 2):
                if (intersect[i+2]-intersect[i+1] > 1):
                    return (int((intersect[i+1]+intersect[i])/2), row)
        return floodFill_start(img, next_row)
    else:
        return floodFill_start(img, next_row)


def floodFill_(img):
    img = np.pad(img, 1, 'constant', constant_values=(0))
    start_ind = floodFill_start(img, 0)
    if not start_ind:
        img[:,:] = 0
        print('NO START')
    else:
        img = img.astype('uint8')
        img = floodFill(img, None, start_ind, 1.0)[1]
        cond = np.where((img==0) | (img==1))
        img[cond] = 1-img[cond]
    img = img.astype('float32')
    #img[np.where(img > 1)] = img[np.where(img > 1)]/255.0
    img = img[1:-1, 1:-1]
    return img


def get_spline_img(p, batch_size, n):
    #time_start = time.time()
    d = 3
    nd = n+d
    rows = 256
    cols = 256

    img_batch = np.zeros((batch_size, rows, cols))

    # Iterate over all examples of batch
    for i in range(batch_size):
        p_i = p[i, :].reshape(n, 2)
        p_temp = np.zeros((nd,2))
        p_temp[:n] = p_i
        p_temp[n:] = p_i[:d]
        p_i = p_temp

        t = np.linspace(0, 1, nd+d+1)
        delta_t = 1/(nd+d)
        spline = BSpline.construct_fast(t, p_i, d, 'periodic')

        # Calculate step size
        step = calc_stepsize(nd, d, p_i, delta_t)

        # Calculate spline
        img = np.zeros((rows, cols))
        t_n = np.linspace(0, 1, int(1.0/step))

        # Find Interior Pixels
        point_prev = spline(t_n[0])
        col_prev= int(point_prev[0])
        row_prev = int(point_prev[1])
        for x in t_n:
            point = spline(x)
            col = int(point[0])
            row = int(point[1])
            if (col==col_prev and row==row_prev):
                continue
            else:
                if img[row_prev, col_prev] == 0:
                    img[row_prev, col_prev] = 1
                col_prev = col
                row_prev = row
        point = spline(t[-1])
        col = int(point[0])
        row = int(point[1])
        img[row, col] = 1

        # Flood fill spline
        img = floodFill_(img)


        # Calculate edge pixel value
        point_prev = spline(t_n[0])
        col_prev = int(point_prev[0])
        row_prev = int(point_prev[1])
        x_arr = []
        y_arr = []
        for x in t_n:

            point = spline(x)
            col = int(point[0])
            row = int(point[1])

            if (col==col_prev and row==row_prev):
                x_arr.append(point[0]-col)
                y_arr.append(point[1]-row)

            else:


                # NORTH
                if img[row_prev-1, col_prev] == 1:
                    img[row_prev, col_prev] = np.abs(scipy.integrate.trapz(y_arr, x_arr))
                # EAST
                elif img[row_prev, col_prev-1] == 1:
                    img[row_prev, col_prev] = np.abs(scipy.integrate.trapz(x_arr, y_arr))
                # SOUTH
                elif img[row_prev+1, col_prev] == 1:
                    img[row_prev, col_prev] = 1 - np.abs(scipy.integrate.trapz(y_arr, x_arr))
                # WEST
                elif img[row_prev, col_prev+1] == 1:
                    img[row_prev, col_prev] = 1 - np.abs(scipy.integrate.trapz(x_arr, y_arr))
                # NORTH WEST
                elif img[row_prev-1, col_prev+1] == 1:
                    img[row_prev, col_prev] = np.abs(scipy.integrate.trapz(y_arr, x_arr))
                # NORTH EAST
                elif img[row_prev-1, col_prev-1] == 1:
                    img[row_prev, col_prev] = np.abs(scipy.integrate.trapz(y_arr, x_arr))
                # SOUTH WEST
                elif img[row_prev+1, col_prev+1] == 1:
                    img[row_prev, col_prev] = 1-np.abs(scipy.integrate.trapz(y_arr, x_arr))
                # SOUTH EAST
                elif img[row_prev+1, col_prev-1] == 1:
                    img[row_prev, col_prev] = 1-np.abs(scipy.integrate.trapz(y_arr, x_arr))


                x_arr = []
                y_arr = []
                x_arr.append(point[0]-col)
                y_arr.append(point[1]-row)
                col_prev = col
                row_prev = row


        # Flood fill spline
        img_batch[i, :, :] = img
        #plt.imshow(img)
        #plt.show()

    return img_batch
