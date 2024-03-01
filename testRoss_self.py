import simulateProcesses as sp
import numpy as np
import pandas as pd
import CoMOGPCCM as gp
import time

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from torch.multiprocessing import Pool
import torch.multiprocessing as mp
import torch
import threading
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def tester(x):
    cuda = 0
    coup = x[0]
    x = x[1]

    data = {
        'name': ['X1-Y', 'X2-Y', 'X3-Y'],
        'x': [x[0], x[1], x[2]],
        'y': [x[3:], x[3:], x[3:]]
    }
    input = pd.DataFrame(data)

    data2 = {
        'name': ['Y1-X', 'Y2-X', 'Y3-X'],
        'y': [x[3], x[4], x[5]],
        'x': [x[:3], x[:3], x[:3]]
    }
    input2 = pd.DataFrame(data2)
    ret = []
    ret2 = []

    GP = gp.GP()
    GP2 = gp.GP()
    ret = GP.testStateSpaceCorrelation(input.iloc[:,1], input.iloc[:,2], 13, tau=2, cuda=cuda)
    ret = torch.stack(ret)
    print(ret)
    ret2 = GP2.testStateSpaceCorrelation(input2.iloc[:,1], input2.iloc[:,2], 13, tau=2, cuda=cuda)
    ret2 = torch.stack(ret2)
    print(ret2)
    print(ret[-1].shape, ret2[-1].shape, coup)

    print("Finish Opti" + str(coup))

    ret = np.array(ret)
    ret2 = np.array(ret2)

    print(ret.shape, ret2.shape, coup)
    t = [[] for i in range(3)]
    for i in range(3):
        for j in range(3):
            t[i] += [np.tanh(1 / 3000 * (ret[i,j][:, None] - ret2[j,i]).ravel())]
    print(t)
    print(coup, np.array(t)[:, :, 0])
    return np.array(t)



if __name__ == '__main__':
   time_start = time.time()
   coups = [0, 0.133, 0.266, 0.4]
   res = []



   #for i in range(30):
   for i in range(1):
       print(i)
       x0 = np.array(sp.rosslerDrivesLorenz(N=3000,dnoise=1e-5, h=.1, eps=.0, initial=np.random.randn(6))) + 1*np.random.randn(6,3000)
       x1 = np.array(sp.rosslerDrivesLorenz(N=3000,dnoise=1e-5, h=.1, eps=.133, initial=np.random.randn(6))) + 1*np.random.randn(6,3000)
       x3 = np.array(sp.rosslerDrivesLorenz(N=3000,dnoise=1e-5, h=.1, eps=.266, initial=np.random.randn(6))) + 1*np.random.randn(6,3000)
       x6 = np.array(sp.rosslerDrivesLorenz(N=3000,dnoise=1e-5, h=.1, eps=.4, initial=np.random.randn(6))) + 1*np.random.randn(6,3000)
       print(x1.shape)

       with Pool(4) as p:
           res +=[p.map(tester,[[1,x1]])]

   res = np.array(res)

   print(res[0])
   label = [[] for i in range(3)]
   for i in range(3):
       for j in range(3):
           label[i] += ["Y{0:d}-X{1:d}".format(i, j)]


   df = pd.DataFrame(columns=['label', 'coup', 'K','pval'])

   for a in range(len(res)):
       for i, r in enumerate(res[a]):
           for j in range(len(r)):
               for k in range(len(r[0])):
                   rr = r[j, k]
                   rasort = np.argsort(rr)
                   CDF = np.arange(len(rasort)) / len(rasort)
                   pval = 1 - CDF[rasort == 0].squeeze()
                   df = df.append({'label': label[j][k], 'coup': format(coups[i]), 'K': rr[0],'pval': pval}, ignore_index=True)
                   plt.plot(rr[rasort], CDF)
                   plt.axvline(rr[0], color="red")
                   plt.xlim((-1, 1))
                   plt.xlabel(r"$\kappa$")
                   plt.ylabel(r"$P(\tilde \kappa)$")
                   plt.title( r"Coupling $\epsilon = {0:.2f}$ pval={1:.3f}".format(coups[i], 1 - CDF[rasort == 0].squeeze()))
                   plt.savefig("rossCoupling{0:.2f}".format(coups[i]) + "_" + label[j][k] + ".png")
                   plt.close()

  # df.to_excel("resultR5.xlsx", index=False)

   time_end = time.time()
   time_sum = time_end - time_start
   print(time_sum)
