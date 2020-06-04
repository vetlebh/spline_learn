
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import BSpline
from cv2 import floodFill



class Spline:


    def __init__(self, p, d=3, clamped=False, closed=True, rows=256, cols=256):
        n = int(p.shape[0]/2)
        self.p = p.reshape((n, 2))
        self.n = n
        self.d = d
        self.clamped = clamped
        self.closed = closed
        self.extrapolate = 'periodic'

        self.rows = rows
        self.cols = cols

        if closed:
            p = np.zeros((n+d,2))
            p[:n] = self.p
            p[n:] = self.p[:d]
            self.p = p
            n = n+d
            self.n = n

        if clamped:
            t = np.zeros(n+d+1)
            t[d:-d] = np.linspace(0, 1, n-d+1)
            t[-d:] = 1
        else:
            t = np.linspace(0, 1, n+d+1)

        self.t = t
        self.spline = BSpline(self.t, self.p, self.d, self.extrapolate)


    def update_p(self, p):
        self.p = p


    def calc_stepsize(self):
         # Calculate step size
        """
        Based on theroretical estimate, see report
        """
        step = 1.0/((self.n+self.d)*np.max(np.sum(self.p, axis=0)))
        return step


    def calc_intersect(self, point):
        cell_height = point[1]-int(point[1])
        return cell_height*255.0


    def floodFill_start(img, row, c):
        """
        Iterates outwards from start row to find point satisfied by
        ray-casting algortihm to be inside spline
        """
        if (row >= 256 or row < 0):
            return False
        next_row = int(row + 10*c)
        c_abs = np.abs(c)
        c = (-1)**c_abs * (c_abs+1)
        intersect = np.where(img[row, :] > 0.0)[0]
        nr_intersect = intersect.shape[0]
        if (nr_intersect > 1):
            for i in range (0, nr_intersect-1, 2):
                if (intersect[i+1]-intersect[i] > 1):
                    return (int((intersect[i+1]+intersect[i])/2), row)
            return floodFill_start(img, next_row, c)
        else:
            return floodFill_start(img, next_row, c)


    def pixelate(self):
        img = np.zeros((self.rows, self.cols))
        stepsize = self.calc_stepsize()
        t = np.linspace(0, 1, int(1.0/stepsize))
        for x in t:
            point = self.spline(x)
            col = int(point[0])
            row = int(point[1])
            if img[row, col] == 0:
                img[row, col] = self.calc_intersect(point)
        self.img = img


    def flood(self, index):
        img = self.img.astype('uint8')
        start_ind = floodFill_start(img, 128, 1)
        if not start_ind:
            self.img[:,:] = 0
        else:
            self.img = floodFill(img, None, start_ind, 1.0)[1]


    def drawCurve(self, steps=100):
        t = np.linspace(0, 1, 100)
        curveX = []
        curveY = []
        for x in t:
            point = self.spline(x)
            curveX.append(point[0])
            curveY.append(point[1])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(curveX, curveY, color='g')
        for i in self.p:
            plt.scatter(i[0], i[1], color='r')
        major_ticksX = np.arange(int(min(curveX)), int(max(curveX)), 1)
        major_ticksY = np.arange(int(min(curveY)), int(max(curveY)), 1)
        ax.set_xticks(major_ticksX)
        ax.set_yticks(major_ticksY)
        ax.grid(which='both')
        plt.show()


    def drawImg(self):
        plt.figure()
        plt.imshow(self.img, origin='lower')
        plt.show()
