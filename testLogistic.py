import numpy as np
import pandas as pd
import cmvgpCCM as gp
import time
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from torch.multiprocessing import Pool
import torch.multiprocessing as mp
import torch
import threading
from numpy.random import multivariate_normal as mnorm
from math import log
from scipy.spatial.distance import pdist, squareform
import preferences as pr

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
   time_start = time.time()


   def logistic_map_coupled(r_x, r_y, x0, y0, eta_x, eta_y, beta_xy, beta_yx, p, e_x, e_y, n):
       """
       Calculate The coupled system state sequence of The logistic map
       :param r_x: Control parameters of the x equation
       :param r_y: Control parameters of the y equation
       :param x0: The initial state value of the x equation
       :param y0: The initial state value of the y equation
       :param eta_x: Coupling parameters of the x equation
       :param eta_y: Coupling parameters of the y equation
       :param beta_xy: The influence coefficient of x equation on y equation
       :param beta_yx: The influence coefficient of y equation on x equation
       :param p: cycle
       :param e_x: Noise signal standard deviation of x equation
       :param e_y: Noise signal standard deviation of y equation
       :param n: State sequence length
       :return: The state sequence of the x equation and the y equation
       """
       x = [x0]
       y = [y0]
       for i in range(n - 1):
           #Ht = np.cos(2 * np.pi * i / p)
           Ht = np.cos(2 * np.pi * i / p)
           x_next = x[i] * ((r_x + eta_x * Ht) * (1 - x[i]) - beta_xy * y[i]) + np.random.normal(0, e_x)
           y_next = y[i] * ((r_y + eta_y * Ht) * (1 - y[i]) - beta_yx * x[i]) + np.random.normal(0, e_y)

           x.append(x_next)
           y.append(y_next)

           return x, y
   r_x = 3.8
   r_y = 3.5
   x0 = 0.2
   y0 = 0.3
   eta_x = 0
   eta_y = 0
   beta_xy = 0
   beta_yx = 0.1
   p = 10
   e_x = 0
   e_y = 0
   n = 1000
   coups = list(range(50, 1300, 50))
   corr1 =[]
   pval = []
   pval2 = []
   res1 =[]
   res2 =[]
   corr2 = []
   df = pd.DataFrame(columns=['coup', 'times', 'CMVGP-CCM', 'pval'])

   for i in range(len(coups)):
       for j in range(10):
           print(i)
           x, y = logistic_map_coupled(r_x, r_y, x0, y0, eta_x, eta_y, beta_xy, beta_yx, p, e_x, e_y, coups[i])
           theta = 0.5
           tau = pr.lag_select(np.squeeze(y), theta)
           print("Estimated embedding lag (tau):", tau)
           Qy = pr.falsenearestneighbors(np.squeeze(y), tau, 0.01, 10)
           print(Qy)
           taux = pr.lag_select(np.squeeze(x), theta)
           print("Estimated embedding lag (tau):", taux)
           Qx = pr.falsenearestneighbors(np.squeeze(x), taux, 0.01, 10)
           print(Qx)

           data = np.vstack((x, y))
           res = pr.tester_self([coups[i], data], tau, Qy, taux, Qx)
           res = np.array(res)
           corr, pval = pr.pval_self(res)
           print(corr, pval)
           #print('======================================')
           df = df.append({'coup': format(coups[i]), 'times': j, 'CMVGP-CCM': corr, 'pval': pval}, ignore_index=True)


   df.to_excel("result_lengthxy.xlsx", index=False)

   time_end = time.time()
   time_sum = time_end - time_start
   print(time_sum)
