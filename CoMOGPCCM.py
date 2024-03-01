import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import torch.nn as nn
import pywt

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


torch.set_default_tensor_type(torch.cuda.FloatTensor)



class GP():
    def __init__(self):
     self.trained = False
     self.cuda = 0
     self.ld = [[] for i in range(3)]
     self.sld = [[] for i in range(3)]
     self.sigma = [[] for i in range(3)]
     self.sigma_l = [[] for i in range(3)]
     self.ssigma = [[] for i in range(3)]
     self.ssigma_l = [[] for i in range(3)]
     self.noise = [[] for i in range(3)]
     self.A = [[] for i in range(3)]
     self.ipoints = [[] for i in range(3)]
     self.hnoise = [[] for i in range(3)]
     self.sipoints = [[] for i in range(3)]
     return

    def setcuda(self, cuda):
         self.cuda = cuda

    def lle(self, X):
        scaler = StandardScaler()

        llee = LocallyLinearEmbedding(n_neighbors=10, n_components=10)
        embedding = llee.fit_transform(X.cpu().numpy())
        embedding = scaler.fit_transform(embedding)
        embedding = torch.from_numpy(embedding).float().cuda(self.cuda)

        return embedding


    def testStateSpaceCorrelation(self, X, Y, m=3, tau=1, cuda=0):
        size = len(X)
        if not X is torch.Tensor:
            for i in range(size):
                X[i] = torch.from_numpy(X[i]).float().cuda(cuda)
                Y[i] = torch.from_numpy(np.array(Y[i])).float().cuda(cuda).T

        # Standardize
        x = []
        y = []
        xar = []
        yar = []
        yp = []
        xp = []
        xi = []

        for i in range(size):
            x.append((X[i] - X[i].mean()) / X[i].std())
            y.append((Y[i] - Y[i].mean(0)) / Y[i].std(0))

            y[i] = y[i].T
            # State space transform
            xar.append(torch.stack([x[i][j:j - tau * m] for j in range(m)]).cuda(cuda).T)
            xar[i] = (xar[i] - xar[i].mean(0)) / xar[i].std(0)
            yar.append(torch.stack([y[i][:, j:j - tau * m] for j in range(m)], 1).cuda(cuda).transpose(1, 2))


            # Train outs
            yp.append(yar[i][:, 1:, :])
            xp.append(xar[i][1:, :])
            xi.append(xar[i][:-1])
            yar[i] = yar[i][:, :-1]

            # Bayesian Regression
            mx = []
            kx = [[] for j in range(len(xar))]

        gpx = self
        gpx.setcuda(cuda)
        for i in range(len(yar[0])):
            yi_list = [yar[j][i] for j in range(len(yar))]
            yy_list = [yp[j][i] for j in range(len(yp))]
            k = gpx.forward(xi, xp, yi_list, yy_list)
            for j in range(len(kx)):
                kx[j].append(k[j].cpu().numpy())

        kx = [np.array(kx[i]).transpose(0,2,1) for i in range(len(kx))]
        kx = [kx[i].astype(float) for i in range(len(kx))]
        kx = [torch.from_numpy(kx[i]) for i in range(len(kx))]
        result = []
        for i in range(len(kx)):
            result += [kx[i]]
        return result

    def squaredExpKernelARD(self, a, b, ld=3):
        r = torch.zeros(1, a.shape[0], b.shape[0]).cuda(self.cuda)

        for i in range(len(ld)):
            temp = torch.cdist(a[None, :, [i]], b[None, :, [i]])
            temp1 = torch.cdist(a[None, :, [i]], b[None, :, [i]],p=1)
            r += temp ** 2 / ld[i]
        cmat = torch.exp(-r)
        return cmat.T.squeeze()



    def latentmade(self, a):
        specific_recon = []
        latent_recon = []
        specific = []
        latent = []
        for i in range(a[0].shape[1]):
            a_all = torch.stack([a[j][:,i] for j in range(len(a))]).T.detach().cpu().numpy()
            pca = PCA(n_components=a_all.shape[1])
            pca.fit(a_all)
            a_common = pca.transform(a_all)
            a_specific = a_all - a_common.dot(pca.components_)
            n = a_all.shape[1]
            w_common = np.corrcoef(a_common.T, a_all.T)[:n, n:]
            w_specific = np.corrcoef(a_specific.T, a_all.T)[:n, n:]
            a_common_recon = a_common.dot(w_common.T)
            a_specific_recon = a_specific.dot(w_specific.T)
            specific_recon += [a_specific_recon]
            latent_recon += [a_common_recon]
        specific_recon = [torch.from_numpy(specific_recon[j]).float().cuda(self.cuda) for j in range(len(specific_recon))]
        latent_recon = [torch.from_numpy(latent_recon[j]).float().cuda(self.cuda) for j in range(len(latent_recon))]

        for i in range(len(a)):
            specific += [torch.stack([specific_recon[j][:,i] for j in range(len(specific_recon))]).T]
            latent += [torch.stack([latent_recon[j][:, i] for j in range(len(latent_recon))]).T]
        return latent, specific




    def covMatrix(self, a, ild, ihnoise, isigma_llist,isigma_list, kernel=None):
        if kernel is None:
            kernel = self.squaredExponentialARD
        cov_matrix = []

        latent,specific = self.latentmade(a)
        latent_cov_matrix_list = [isigma_llist[i]*kernel(latent[i], latent[i], ild[i])+(ihnoise[i]**2).diag() for i in range(len(latent))]

        # covariance matrix list
        special_cov_matrix_list = [isigma_list[i]*kernel(specific[i], specific[i], ild[i]) + (ihnoise[i]**2).diag() for i in range(len(specific))]
        ori_cov_matrix_list = [isigma_list[i] * kernel(a[i], a[i], ild[i]) + (ihnoise[i] ** 2).diag() for i in range(len(a))]

        for i in range(len(special_cov_matrix_list)):
            cov_matrix.append((latent_cov_matrix_list[i] * 0.1 + special_cov_matrix_list[i]) * 0.1 + ori_cov_matrix_list[i] * 0.8)
        return cov_matrix

    def covMatrixnn(self, a, ild, isigma_llist,isigma_list, kernel=None):
        if kernel is None:
            kernel = self.squaredExponentialARD
        cov_matrix = []
        latent, specific = self.latentmade(a)
        # linear combination covariance matrix of each objective function
        latent_cov_matrix_list = [isigma_llist[i] * kernel(latent[i], latent[i], ild[i]) for i in range(len(latent))]

        # covariance matrix list
        special_cov_matrix_list = [isigma_list[i]*kernel(specific[i], specific[i], ild[i]) for i in range(len(specific))]
        # special correlation matrix
        ori_cov_matrix_list = [isigma_list[i] * kernel(a[i], a[i], ild[i]) for i in range(len(a))]

        for i in range(len(special_cov_matrix_list)):
            cov_matrix.append((latent_cov_matrix_list[i] * 0.1 + special_cov_matrix_list[i]) * 0.1 + ori_cov_matrix_list[i] * 0.8)
        return cov_matrix

    def optimizeHyperparms(self, data, inp, lr=.001, ld=[[1.],[1.],[1.]], sigma_l =[[6.],[6.],[6.]],sigma =[[6.],[6.],[6.]], noise=[[.1],[.1],[.1]], niter=20, ns=False, kernel=None, m=40):
        for i in range(len(data)):
            data[i].requires_grad_(False)
            inp[i].requires_grad_(False)
        if kernel is None:
            kernel = self.squaredExponentialARD
        data = (data)
        ip = inp
        train = data

        # Define Hyperparameters and their distributions
        A = [(torch.randn(inp[i].shape[-1], inp[i].shape[-1])).float().cuda(self.cuda).requires_grad_() for i in range(len(inp))]
        V = [torch.linalg.svd(A[i])[-1] for i in range(len(A))]
        sigma_l = [torch.tensor(1 * sigma_l[i]).float().log().cuda(self.cuda).requires_grad_() for i in range(len(sigma_l))]
        sigma = [torch.tensor(1 * sigma[i]).float().log().cuda(self.cuda).requires_grad_() for i in range(len(sigma))]
        noiseo = [torch.tensor(noise[i]).float().cuda(self.cuda).requires_grad_() for i in range(len(noise))]
        hnoise = [(noise[i] * (torch.ones(m).float())).cuda(self.cuda).requires_grad_() for i in range(len(noise))]
        sld = [(1 * torch.ones(inp[i].shape[-1])).float().cuda(self.cuda).requires_grad_() for i in range(len(inp))]
        ld = [(ld[i] * torch.ones(inp[i].shape[-1])).float().log().cuda(self.cuda).requires_grad_() for i in range(len(ld))]
        ssigma_l = [(1 * torch.ones(1)).float().cuda(self.cuda).requires_grad_() for i in range(len(inp))]
        ssigma = [(1 * torch.ones(1)).float().cuda(self.cuda).requires_grad_() for i in range(len(inp))]

        # Get some nice inducing points
        p = [np.random.permutation(len(data[i]))[:m] for i in range(len(data))]
        inp = [ip[i].matmul(V[i]) for i in range(len(V))]
        sipoints = [(1 * torch.ones(*inp[i][p[i]].shape)).cuda(self.cuda).requires_grad_() for i in range(len(inp))]
        ipoints = [torch.clone(inp[i][p[i]].detach()).cuda(self.cuda).requires_grad_() for i in range(len(inp))]
        msigma = []
        mipoints = []
        mld = []

        for i in range(len(inp)):
            msigma.append(sigma[i].detach().requires_grad_(False))
            mipoints.append(ipoints[i].detach().requires_grad_(False))
            mld.append(ld[i].detach().requires_grad_(False))


        # put hyperparms we want to gradclip in list
        parms =[[sigma[i], ssigma[i],sigma_l[i],ssigma_l[i], ld[i], sld[i], ipoints[i], sipoints[i]] for i in range(len(sigma))]

        mSamples = 5
        eye = [torch.eye(len(inp[i])).cuda(self.cuda).float() for i in range(len(inp))]
        for i in range(niter):
            ls = []
            for k in range(mSamples):
                # Sample hyperparameters
                V = [torch.linalg.svd(A[i])[-1] for i in range(len(A))]
                inp = [ip[i] @ V[i] for i in range(len(ip))]
                ripoints = [(ipoints[i] + (sipoints[i] * torch.randn(*ipoints[i].shape).cuda(self.cuda))) for i in range(len(ipoints))]
                rld = [(ld[i] + (sld[i] * torch.randn(*ld[i].shape).cuda(self.cuda))).exp() for i in range(len(ld))]
                rsigma_l = [(sigma_l[i] + (ssigma_l[i] * torch.randn(1).cuda(self.cuda))).exp() for i in range(len(sigma_l))]
                rsigma = [(sigma[i] + (ssigma[i] * torch.randn(1).cuda(self.cuda))).exp() for i in range(len(sigma))]


                # Sparse kernel GP likelihood log likelihood
                Km = self.covMatrix(ripoints, rld, hnoise,rsigma_l, rsigma, kernel=kernel)
                Kn = self.covMatrixnn(inp, rld, rsigma_l, rsigma, kernel=kernel)
                Kmi = [torch.inverse(Km[i]) for i in range(len(Km))]
                Knm = [rsigma[i] * kernel(inp[i], ripoints[i], rld[i]) for i in range(len(inp))]
                lamb = [(Kn[i] - Knm[i].T @ Kmi[i] @ Knm[i]).diag().diag() for i in range(len(Kn))]
                K = [Knm[i].T @ Kmi[i] @ Knm[i] + lamb[i] + noiseo[i] ** 2 * eye[i] for i in range(len(Knm))]
                Kinv = [torch.inverse(K[i]) for i in range(len(K))]
                logp = [0 for i in range(3)]
                trr = [train[i] @ V[i] for i in range(len(V))]
                for i in range(len(trr)):
                    for tr in trr[i].T:
                        logp[i] += -.5 * tr.T @ Kinv[i] @ tr
                    logp[i] = (logp[i] / train[i].shape[-1] - torch.linalg.cholesky(K[i]).slogdet()[1]).mean()
                logp = [(logp[i] / mSamples) for i in range(len(logp))]
                logp = torch.stack(logp)
                logp.backward(torch.ones_like(logp))
                ls += [logp[i].item() for i in range(len(logp))]

            # KL Divergence between approximate posterior and prior distribution
            logp_list = []
            for i in range(len(sigma)):
                logp = -1 / 2 * ((-msigma[i] + sigma[i]) ** 2 - 1 + ssigma[i] ** 2) - torch.log(ssigma[i] / 1)
                logp -= (1 / 2 * ((-mld[i] + ld[i]) ** 2 - 1 + sld[i] ** 2) - torch.log(sld[i] / 1)).sum()
                logp -= (1 / 2 * ((-mipoints[i] + ipoints[i]) ** 2 - 1 + sipoints[i] ** 2) - torch.log(sipoints[i] / 1)).sum()

                logp_list.append(logp)

            logp = torch.stack(logp_list)

            # Autograd
            logp.backward(torch.ones_like(logp))

            # Grad clip
            for i in range(len(parms)):
                torch.nn.utils.clip_grad_norm_(parms[i], 1000)

            # Gradient Ascent
            with torch.no_grad():
                for i in range(len(sigma)):
                    sigma[i] += lr * sigma[i].grad
                    sigma_l[i] += lr * sigma_l[i].grad
                    ld[i] += lr * ld[i].grad
                    noiseo[i] += lr * noiseo[i].grad
                    hnoise[i] += lr * hnoise[i].grad
                    ipoints[i] += lr * ipoints[i].grad
                    ssigma[i] += lr * ssigma[i].grad
                    ssigma_l[i] += lr * ssigma_l[i].grad
                    sld[i] += lr * sld[i].grad
                    sipoints[i] += lr * sipoints[i].grad
                    A[i] += lr * A[i].grad

            # Zero out calculated grad
            for i in range(len(sigma)):
                sigma[i].grad.zero_()
                sigma_l[i].grad.zero_()
                ld[i].grad.zero_()
                noiseo[i].grad.zero_()
                hnoise[i].grad.zero_()
                ipoints[i].grad.zero_()
                A[i].grad.zero_()

        for i in range(len(ld)):
            self.ld[i] = ld[i].detach().requires_grad_(False)
            self.sld[i] = sld[i].detach().requires_grad_(False)

            self.sigma[i] = sigma[i].detach().requires_grad_(False)
            self.sigma_l[i] = sigma_l[i].detach().requires_grad_(False)
            self.ssigma[i] = ssigma[i].detach().requires_grad_(False)
            self.ssigma_l[i] = ssigma_l[i].detach().requires_grad_(False)

            self.noise[i] = noiseo[i].detach().requires_grad_(False) ** 2
            self.A[i] = V[i].detach().requires_grad_(False)
            self.hnoise[i] = hnoise[i].detach() ** 2
            self.ipoints[i] = ipoints[i].detach().requires_grad_(False)
            self.sipoints[i] = sipoints[i].detach().requires_grad_(False)

        return

    # Posterior Inference
    def forward(self, inp, observations, test, testo, ld=[[1],[1],[1]], sig=10, noise=[[.1],[.1],[.1]], target="squaredexp"):
        # Choose kernel
        kernel = self.squaredExpKernelARD

        ltest = test

        # Get Hyperparms
        if self.trained is False:
            noise = np.ones(len(inp))

            sigma_l = [inp[i].std().item() ** (1 / 2) for i in range(len(inp))]
            sigma = [inp[i].std().item() ** (1 / 2) for i in range(len(inp))]
            ld = [(inp[i].std().item()) ** (1 / 2) for i in range(len(inp))]
            self.optimizeHyperparms(observations, inp, sigma_l=sigma_l,sigma=sigma, noise=noise, ld=ld, niter=40, lr=1e-4,kernel=kernel)
            self.trained = True

        posteriorK = [[] for i in range(len(inp))]
        ltest = [ltest[i].matmul(self.A[i]) for i in range(len(ltest))]
        # Build Null distribution
        for i in range(30):
            if i == 0:
                t = 0
            else:
                t = 1
                p = [np.random.permutation(len(ltest[i].ravel())) for i in range(len(ltest))]
                ltest = [ltest[i].reshape(-1)[p[i]].reshape(*ltest[i].shape).cuda(self.cuda) for i in range(len(ltest))]
            sigma_l = [(self.sigma_l[i] + (self.ssigma_l[i] * np.random.randn() * t)).exp() for i in range(len(self.sigma_l))]
            sigma = [(self.sigma[i] + (self.ssigma[i] * np.random.randn() * t)).exp() for i in range(len(self.sigma))]
            ld = [(self.ld[i] + (self.sld[i] * torch.randn(*self.ld[i].shape).cuda(self.cuda) * t)).exp() for i in range(len(self.ld))]
            ipoints =[self.ipoints[i] + (self.sipoints[i] * torch.randn(*self.ipoints[i].shape).cuda(self.cuda) * t) for i in range(len(self.ipoints))]
            tk = [sigma[i] * kernel(ltest[i], ltest[i], ld[i]) for i in range(len(ltest))]

            # Test given train kernel
            ip = [inp[i].matmul(self.A[i]) for i in range(len(inp))]
            inptk = [sigma[i] * kernel(ltest[i], ipoints[i], ld[i]) for i in range(len(ltest))]
            Km = self.covMatrix(ipoints, ld, self.hnoise, sigma_l, sigma, kernel=kernel)
            Knm = [sigma[i] * kernel(inp[i], ipoints[i], ld[i]) for i in range(len(inp))]
            Kmi = [torch.inverse(Km[i]) for i in range(len(Km))]
            Kn = self.covMatrixnn(ip, ld,sigma_l, sigma, kernel=kernel)
            lamb = [(Kn[i] - Knm[i].T.matmul(Kmi[i]).matmul(Knm[i])).diag().diag() for i in range(len(Knm))]
            eye = [torch.eye(tk[i].shape[-1]).cuda(self.cuda) for i in range(len(tk))]
            lni = [(1 / (lamb[i] + torch.tensor(noise[i]) * eye[i]).diag()).diag() for i in range(len(lamb))]
            Qm = [Km[i] + Knm[i].matmul(lni[i]).matmul(Knm[i].T) for i in range(len(Km))]
            stdMat = [torch.inverse(Km[i]) - torch.inverse(Qm[i]) for i in range(len(Km))]

            # Covariance update
            for i in range(len(posteriorK)):
                posteriorK[i].append([(tk[i] - inptk[i].T.matmul(stdMat[i]).matmul(inptk[i]) + self.noise[i] * eye[i]).slogdet()[1].cpu()])

        for i in range(len(posteriorK)):
            posteriorK[i] = [torch.tensor(ks) for ks in posteriorK[i]]
            posteriorK[i] = torch.stack(posteriorK[i])

        return posteriorK



