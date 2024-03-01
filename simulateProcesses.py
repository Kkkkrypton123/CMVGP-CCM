import numpy as np

# Some nice complex signals to simulate
def lorenzGenerator(initial_state=[1,1,1], eps=1, N=10000, sigma=10, beta=8/3, rho=28, noise=1e-1, h=1e-2):
    x = np.zeros(N,dtype=float)
    y = np.zeros(N,dtype=float)
    z = np.zeros(N,dtype=float)
    initial_state = np.array(initial_state)
    x[0] = initial_state[0]
    y[0] = initial_state[1]
    z[0] = initial_state[2]

    def dxdydz(xt, yt, zt):
        dx = sigma*(yt - xt)
        dy = (xt)*(rho - zt) - yt
        dz = xt*yt - beta*zt
        return np.array([dx, dy, dz])

    for i, (xt,yt,zt) in enumerate(zip(x[:-1], y[:-1], z[:-1])):
        # Runge Kutta integration
        k1 = h*dxdydz(xt,yt,zt)
        ink2 = (xt + k1[0]/2, yt + k1[1]/2, zt + k1[2]/2,)
        k2 = h*dxdydz(*ink2)
        ink3 = (xt + k2[0]/2, yt + k2[1]/2, zt + k2[2]/2,)
        k3 = h*dxdydz(*ink3)
        ink4 = (xt + k3[0], yt + k3[1], zt + k3[2],)
        k4 = h*dxdydz(*ink4)
        x[i+1] = xt + 1/6*(k1[0] + k2[0]*2 + 2*k3[0] + k4[0]) + np.random.randn()*noise
        y[i+1] = yt + 1/6*(k1[1] + k2[1]*2 + 2*k3[1] + k4[1]) + np.random.randn()*noise
        z[i+1] = zt + 1/6*(k1[2] + k2[2]*2 + 2*k3[2] + k4[2]) + np.random.randn()*noise
    return x,y,z

def onefnoise(l=1000, amp=1):
    xf = 1/np.arange(1,l//2+2)*(l//2)*amp
    xfc = xf*np.exp(-2*np.pi*np.random.rand(l//2+1)*1j)
    return np.fft.irfft(xfc)


def henonMapGenerator(initial_state=[1,1], N=10000, alpha=1.4, beta=.3, noise=1e-1):
    x = np.zeros(N,dtype=float)
    y = np.zeros(N,dtype=float)
    x[0] = initial_state[0]
    y[0] = initial_state[1]
    for i, (xt,yt) in enumerate(zip(x[:-1], y[:-1])):
        x[i+1] = 1- alpha*xt**2 + yt + np.random.randn()*noise
        y[i+1] = beta*xt + np.random.randn()*noise
    return x,y

def simulateEEG(alpha, beta, gamma, delta, theta, N=10000):
    aphase = np.random.rand()*2*np.pi
    bphase = np.random.rand()*2*np.pi
    gphase = np.random.rand()*2*np.pi
    dphase = np.random.rand()*2*np.pi
    tphase = np.random.rand()*2*np.pi
    aband = alpha*np.sin([2*np.pi*i/200 * ((14-13)*np.random.rand() + 13) + aphase for i in range(N)])
    bband = beta*np.sin([2*np.pi*i/200 * ((32-28)*np.random.rand() + 28) + bphase for i in range(N)])
    gband = gamma*np.sin([2*np.pi*i/200 * ((45-40)*np.random.rand() + 40) + gphase for i in range(N)])
    dband = delta*np.sin([2*np.pi*i/200 * ((1-.5)*np.random.rand() + .5) + dphase for i in range(N)])
    tband = theta*np.sin([2*np.pi*i/200 * ((8-6)*np.random.rand() + 6) + tphase for i in range(N)])
    return aband + bband + gband + dband + tband

def sugiSig(initial=[.5,.5], N=10000, rx=3.8, ry=3.5, Bxy=.8, Byx=.2, noise=1e-3):
    x = [initial[0]]
    y = [initial[1]]
    for i in range(N):
        x.append(x[-1]*(rx - rx*x[-1] - Bxy*y[-1]) + noise*np.random.randn())
        y.append(y[-1]*(ry - ry*y[-1] - Byx*x[-2]) + noise*np.random.randn())
    return x[1:], y[1:]

def RLcircuit(N=1000, R=5, L=10, ohm=0.01, h=0.01, noise=[0], anoise=0):
    I = np.zeros((len(noise),N),dtype=float).T
    V = np.sin(ohm*np.arange(N)*h) + np.random.randn(N)*anoise
    def didt(I, V):
        di = V/L - R/L*I
        return di
    for p in range(len(noise)):
        for i, (v,cur) in enumerate(zip(V[:-1], I[:-1,p])):
            # Runge Kutta integration
            k1 = h*didt(cur, v)
            ink2 = cur + k1/2
            k2 = h*didt(ink2, v)
            ink3 = cur + k2/2
            k3 = h*didt(ink3,v)
            ink4 = cur + k3
            k4 = h*didt(ink4, v)
            I[i+1,p] = cur + 1/6*(k1 + k2*2 + 2*k3 + k4) + np.random.randn()*noise[p]
    return V, I.squeeze()

def lorenzDrivesRossler(N=1000, eps=.3, ylag=0, w1=1.015, w2=.985, h=.01, dnoise=0, sigma=10, beta=8/3, rho=28, initial=[1,1,1,1,1,1], dynamic=False):
    x1 = np.random.randn(N)*initial[0]
    x2 = np.random.rand(N)*initial[1]
    x3 = np.random.rand(N)*initial[2]

    y1 = np.random.rand(N)*initial[3]
    y2 = np.random.rand(N)*initial[4]
    y3 = np.random.rand(N)*initial[5]

    def dxdt(xt, yt, zt):
        dx = sigma*(yt - xt)# +  eps*(x1[t-ylag] - xt)
        dy = (xt)*(rho - zt) - yt
        dz = xt*yt - beta*zt
        return np.array([dx, dy, dz])

    def dydt(y1t, y2t, y3t, t):
#        dy1 = -w2*y2t - y3t +  eps*(.1*(x1[t-ylag]) - y1t)
        dy1 = -w2*y2t - y3t +  eps*y1t*(.1*x1[t-ylag] - 1)
        dy2 = w2*y1t + .15*y2t
        dy3 = .2 + y3t*(y1t - 10)
        return np.array([dy1, dy2, dy3])

    for i, (x1_t,x2_t,x3_t) in enumerate(zip(x1[:-1], x2[:-1], x3[:-1])):
        if dynamic:
            t = i*h
        else:
            t = 1*h
        # Runge Kutta integration
        k1 = h*dxdt(x1_t,x2_t,x3_t)
        ink2 = (x1_t + k1[0]/2, x2_t + k1[1]/2, x3_t + k1[2]/2,)
        k2 = h*dxdt(*ink2)
        ink3 = (x1_t + k2[0]/2, x2_t + k2[1]/2, x3_t + k2[2]/2,)
        k3 = h*dxdt(*ink3)
        ink4 = (x1_t + k3[0], x2_t + k3[1], x3_t + k3[2],)
        k4 = h*dxdt(*ink4)
        x1[i+1] = x1_t + 1/6*(k1[0] + k2[0]*2 + 2*k3[0] + k4[0]) + t*np.random.randn()*dnoise
        x2[i+1] = x2_t + 1/6*(k1[1] + k2[1]*2 + 2*k3[1] + k4[1]) + t*np.random.randn()*dnoise
        x3[i+1] = x3_t + 1/6*(k1[2] + k2[2]*2 + 2*k3[2] + k4[2]) + t*np.random.randn()*dnoise

    for i, (y1_t,y2_t,y3_t) in enumerate(zip(y1[ylag:-1], y2[ylag:-1], y3[ylag:-1])):
        if dynamic:
            t = i*h
        else:
            t = 1*h

        # Runge Kutta integration
        k1 = h*dydt(y1_t,y2_t,y3_t, ylag+i)
        ink2 = (y1_t + k1[0]/2, y2_t + k1[1]/2, y3_t + k1[2]/2,)
        k2 = h*dydt(*ink2, ylag+i)
        ink3 = (y1_t + k2[0]/2, y2_t + k2[1]/2, y3_t + k2[2]/2,)
        k3 = h*dydt(*ink3, ylag+i)
        ink4 = (y1_t + k3[0], y2_t + k3[1], y3_t + k3[2],)
        k4 = h*dydt(*ink4, ylag+i)
        y1[ylag+i+1] = y1_t + 1/6*(k1[0] + k2[0]*2 + 2*k3[0] + k4[0]) + np.random.randn()*dnoise
        y2[ylag+i+1] = y2_t + 1/6*(k1[1] + k2[1]*2 + 2*k3[1] + k4[1]) + np.random.randn()*dnoise
        y3[ylag+i+1] = y3_t + 1/6*(k1[2] + k2[2]*2 + 2*k3[2] + k4[2]) + np.random.randn()*dnoise
    return x1,x2,x3,y1,y2,y3


def rosslerDrivesLorenz(N=1000, eps=.3, ylag=0, w1=1.015, w2=.985, h=.01, dnoise=0, sigma=10, beta=8/3, rho=28, initial=[1,1,1,1,1,1], dynamic=False):
    x1 = np.random.randn(N)*initial[0]
    x2 = np.random.randn(N)*initial[1]
    x3 = np.random.randn(N)*initial[2]

    y1 = np.random.randn(N)*initial[3]
    y2 = np.random.randn(N)*initial[4]
    y3 = np.random.randn(N)*initial[5]

    def dydt(xt, yt, zt, t):
#        dx = sigma*(yt - xt) +  eps*((x1[t-ylag] - x1.mean()) - xt)
        dx = sigma*(yt - xt) +  eps*xt*(x1[t-ylag] - 1)
        dy = (xt)*(rho - zt) - yt 
        dz = xt*yt - beta*zt
        return np.array([dx, dy, dz])

    def dxdt(y1t, y2t, y3t):
        dy1 = -w2*y2t - y3t# +  eps*(x1[t-ylag] - y1t)
        dy2 = w2*y1t + .15*y2t
        dy3 = .2 + y3t*(y1t - 10)
        return np.array([dy1, dy2, dy3])

    for i, (x1_t,x2_t,x3_t) in enumerate(zip(x1[:-1], x2[:-1], x3[:-1])):
        if dynamic:
            t = i*h
        else:
            t = 1
        # Runge Kutta integration
        k1 = h*dxdt(x1_t,x2_t,x3_t)
        ink2 = (x1_t + k1[0]/2, x2_t + k1[1]/2, x3_t + k1[2]/2,)
        k2 = h*dxdt(*ink2)
        ink3 = (x1_t + k2[0]/2, x2_t + k2[1]/2, x3_t + k2[2]/2,)
        k3 = h*dxdt(*ink3)
        ink4 = (x1_t + k3[0], x2_t + k3[1], x3_t + k3[2],)
        k4 = h*dxdt(*ink4)
        x1[i+1] = x1_t + 1/6*(k1[0] + k2[0]*2 + 2*k3[0] + k4[0]) + t*np.random.randn()*dnoise
        x2[i+1] = x2_t + 1/6*(k1[1] + k2[1]*2 + 2*k3[1] + k4[1]) + t*np.random.randn()*dnoise
        x3[i+1] = x3_t + 1/6*(k1[2] + k2[2]*2 + 2*k3[2] + k4[2]) + t*np.random.randn()*dnoise
#        x1 = np.clip(x1, -100,100)
#        x2 = np.clip(x2, -100,100)
#        x3 = np.clip(x3, -100,100)

    for i, (y1_t,y2_t,y3_t) in enumerate(zip(y1[ylag:-1], y2[ylag:-1], y3[ylag:-1])):
        if dynamic:
            t = i*h
        else:
            t = 1
        # Runge Kutta integration
        k1 = h*dydt(y1_t,y2_t,y3_t, ylag+i)
        ink2 = (y1_t + k1[0]/2, y2_t + k1[1]/2, y3_t + k1[2]/2,)
        k2 = h*dydt(*ink2, ylag+i)
        ink3 = (y1_t + k2[0]/2, y2_t + k2[1]/2, y3_t + k2[2]/2,)
        k3 = h*dydt(*ink3, ylag+i)
        ink4 = (y1_t + k3[0], y2_t + k3[1], y3_t + k3[2],)
        k4 = h*dydt(*ink4, ylag+i)
        y1[ylag+i+1] = y1_t + 1/6*(k1[0] + k2[0]*2 + 2*k3[0] + k4[0]) + t*np.random.randn()*dnoise
        y2[ylag+i+1] = y2_t + 1/6*(k1[1] + k2[1]*2 + 2*k3[1] + k4[1]) + t*np.random.randn()*dnoise
        y3[ylag+i+1] = y3_t + 1/6*(k1[2] + k2[2]*2 + 2*k3[2] + k4[2]) + t*np.random.randn()*dnoise
#        y1 = np.clip(y1, -100,100)
#        y2 = np.clip(y2, -100,100)
#        y3 = np.clip(y3, -100,100)

    return x1,x2,x3,y1,y2,y3


def rosslerAttractor(N=1000, eps=.3, ylag=0, w1=1.015, w2=.985, h=.01, dnoise=0, initial=[1,1,1,1,1,1]):
    x1 = np.ones(N, dtype=float)#*initial[0]
    x2 = np.ones(N, dtype=float)#*initial[1]
    x3 = np.ones(N, dtype=float)#*initial[2]

    y1 = np.ones(N, dtype=float)#*initial[3]
    y2 = np.ones(N, dtype=float)#*initial[4]
    y3 = np.ones(N, dtype=float)#*initial[5]

    def dxdt(x1t, x2t, x3t):
        dx1 = -w1*x2t - x3t
        dx2 = w1*x1t + .15*x2t
        dx3 = 0.2 + x3t*(x1t - 10)
        return np.array([dx1,dx2, dx3])

    def dydt(y1t, y2t, y3t, t):
        dy1 = -w2*y2t - y3t +  eps*(x1[t-ylag] - y1t)
        dy2 = w2*y1t + .15*y2t
        dy3 = .2 + y3t*(y1t - 10)
        return np.array([dy1, dy2, dy3])

    for i, (x1_t,x2_t,x3_t) in enumerate(zip(x1[:-1], x2[:-1], x3[:-1])):
        # Runge Kutta integration
        k1 = h*dxdt(x1_t,x2_t,x3_t)
        ink2 = (x1_t + k1[0]/2, x2_t + k1[1]/2, x3_t + k1[2]/2,)
        k2 = h*dxdt(*ink2)
        ink3 = (x1_t + k2[0]/2, x2_t + k2[1]/2, x3_t + k2[2]/2,)
        k3 = h*dxdt(*ink3)
        ink4 = (x1_t + k3[0], x2_t + k3[1], x3_t + k3[2],)
        k4 = h*dxdt(*ink4)
        x1[i+1] = x1_t + 1/6*(k1[0] + k2[0]*2 + 2*k3[0] + k4[0]) + np.random.randn()*dnoise
        x2[i+1] = x2_t + 1/6*(k1[1] + k2[1]*2 + 2*k3[1] + k4[1]) + np.random.randn()*dnoise
        x3[i+1] = np.clip(x3_t + 1/6*(k1[2] + k2[2]*2 + 2*k3[2] + k4[2]) + np.random.randn()*dnoise, -100,100)

    for i, (y1_t,y2_t,y3_t) in enumerate(zip(y1[ylag:-1], y2[ylag:-1], y3[ylag:-1])):
        # Runge Kutta integration
        k1 = h*dydt(y1_t,y2_t,y3_t, ylag+i)
        ink2 = (y1_t + k1[0]/2, y2_t + k1[1]/2, y3_t + k1[2]/2,)
        k2 = h*dydt(*ink2, ylag+i)
        ink3 = (y1_t + k2[0]/2, y2_t + k2[1]/2, y3_t + k2[2]/2,)
        k3 = h*dydt(*ink3, ylag+i)
        ink4 = (y1_t + k3[0], y2_t + k3[1], y3_t + k3[2],)
        k4 = h*dydt(*ink4, ylag+i)
        y1[ylag+i+1] = y1_t + 1/6*(k1[0] + k2[0]*2 + 2*k3[0] + k4[0]) + np.random.randn()*dnoise
        y2[ylag+i+1] = y2_t + 1/6*(k1[1] + k2[1]*2 + 2*k3[1] + k4[1]) + np.random.randn()*dnoise
        y3[ylag+i+1] = np.clip(y3_t + 1/6*(k1[2] + k2[2]*2 + 2*k3[2] + k4[2]) + np.random.randn()*dnoise, -100,100)
    return x1,x2,x3,y1,y2,y3


def generateCorrelatedNoise(cov=[1, .5, .3], N=10000):
    k = len(cov)
    cov = np.vstack([np.roll(cov,n) for n in range(k)])
    rM = np.random.randn(k,N)
    return np.linalg.cholesky(cov).dot(rM)
