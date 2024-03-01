import numpy as np
import statsmodels.tsa.api as smt
import cmvgpCCM as gp
import torch
import pandas as pd


def lag_select(x, theta):
    """
    Select tau for SSR using the autocorrelation function

    Inputs:
    x = your signal
    theta = cutoff parameter for the autocorrelation function. When ACF(tau) < theta, we select tau as the embedding lag.

    Outputs:
    tau, the estimate of the embedding lag
    """

    N = len(x)
    ACF = smt.stattools.acf(x, nlags = N-1)



    if np.all(ACF >= theta):
        ACF = smt.stattools.acf(x, nlags = N-1)

    tau = np.where(ACF < theta)[0][0]

    return tau


def embed(y, Q, tau, k=0):
    """
    Wrapper script to generate time-delay embedding vectors for SSR

    Inputs:
      y = your signal as a column vector
      Q = embedding dimension
      tau = embedding lag
      k = forecast length, if you're doing that

    Outputs:
      M = the shadow manifold, as an L*Q matrix where L=1+(Q-1)*tau
      t = forecast targets, if you're doing that
    """
    if k is None:
        k = 0
    if y.ndim == 1:
        # 获取y的长度N
        N = len(y)
    else:
        # 获取y的列数N
        N = y.shape[1]
    part1 = np.arange(1, 1 + (Q - 1) * tau + 1, tau)
    part2 = np.arange(1, N - (Q - 1) * tau - 1 - k + 1)
    part1 = np.expand_dims(part1, axis=1)  # 将part1转换为列向量
    part2 = np.expand_dims(part2, axis=1)  # 将part2转换为列向量
    idm = np.tile(part2, (1, len(part1))) + part1.T
    idt = idm[:, -1] + k
    M = y[idm-1]
    t = y[idt-1]
    return M


def falsenearestneighbors(y, tau, FNNtol, Qmax):

    """
    Select embedding dimension using false nearest neighbors criterion

    Inputs:
    y = your signal (as a column vector)
    tau = embedding lag parameter
    FNNtol = tolerance of the false nearest neighbors method, i.e. the fraction of points that are false nearest neighbors that you are willing to accept. Usually set this to something small like 0.01

    Outputs:
    Q = embedding dimension
    """

    if Qmax is None:
        Qmax = 10

    rho = 17
    Q = 1
    FNNflag = False

    while not FNNflag:
        Q = Q + 1

        if Q > Qmax:
            print('FNN algorithm failed to converge. FNN=%0.2f\n Forcing Q=%d.\n' % (np.mean(FNN), Qmax))
            Q = Qmax
            break

        M1 = embed(y, Q, tau)
        M2 = embed(y, Q + 1, tau)

        # Make sure that these guys are the same size
        M1 = M1[:M2.shape[0]]
        FNN = np.zeros(M1.shape[0])

        for n in range(M1.shape[0]):
            _, id = np.argsort(np.linalg.norm(M1 - M1[n, :], axis=1))[:2]
            Rd = np.linalg.norm(M1[id, :] - M1[n, :]) / np.sqrt(Q)
            FNN[n] = np.linalg.norm(M2[n, :] - M2[id, :]) > rho * Rd

        if np.mean(FNN) < FNNtol:
            FNNflag = True
    if Q <= 6:
        #To prevent singular value decomposition failure
        Q = 6

    return Q

def tester(x, lagy, Qy, lagx, Qx):

    cuda = 0
    coup = x[0]
    x = x[1]

    data = {
        'x': [x[0], x[0], x[0]],
        'y': [x[1], x[1], x[1]]
    }
    input = pd.DataFrame(data)
    data2 = {
        'y': [x[1], x[1], x[1]],
        'x': [x[0], x[0], x[0]]
    }
    input2 = pd.DataFrame(data2)
    ret = []
    ret2 = []

    GP = gp.GP()
    GP2 = gp.GP()
    ret = GP.testStateSpaceCorrelation(input.iloc[:, 0], input.iloc[:, 1], Qy, tau=lagy, cuda=cuda)
    ret = torch.stack(ret)

    ret2 = GP2.testStateSpaceCorrelation(input2.iloc[:, 0], input2.iloc[:, 1], Qx, tau=lagx, cuda=cuda)
    ret2 = torch.stack(ret2)

    ret = np.array(ret)
    ret2 = np.array(ret2)

    t = []
    t = [np.tanh(1 / coup * (ret[0][:, None] - ret2[0]).ravel())]

    return np.array(t)

def pval_self(res):
    for i, r in enumerate(res):
        rasort = np.argsort(r)
        CDF = np.arange(len(rasort)) / len(rasort)
        pval = 1 - CDF[rasort == 0].squeeze()
        rr = r[0]

    return rr,pval

