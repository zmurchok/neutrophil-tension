import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint, solve_ivp
from matplotlib import cm

plt.rc("text", usetex=False)
plt.rc("font", **{'family':'sans-serif', 'sans-serif':['Helvetica'], 'size':10})

def f(u, v, b, gamma, n, RT, delta):
    return (b+gamma*u**n/(1+u**n))*v - delta*u

def g(u, v, b, gamma, n, RT, delta):
    return -f(u,v,b, gamma, n, RT, delta)

def F(R):
    sharp = 10
    switch = 1
    magnitude = 0.001
    return magnitude/(1+np.exp(-2*sharp*(R-switch)))

def xmdot(u,v,xm,xp,t):
    viscosity = 1
    spring = 0.01
    return (spring*(xp-xm-1) - F(u[0]))/viscosity

def xpdot(u,v,xm,xp,t):
    viscosity = 1
    spring = 0.01
    return (-spring*(xp-xm-1) + F(u[-1]))/viscosity

def rdPDE(y, t, b0, b1, gamma, n, RT, d0, d1, Du, Dv, dx):
    """
    The ODEs are derived using the method of lines.
    https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#mol
    """
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    #the ode for L is stored at the end of y
    u = y[0:-2:2]
    v = y[1:-2:2]
    xm = y[-2]
    xp = y[-1]

    l = xp-xm

    bvec = b0*np.ones(np.shape(np.linspace(0,1,1/dx))) + b1*(l*np.linspace(0,1,1/dx)+xm)

    delta = d0 + d1*(l-1)

    # dydt is the return value of this function.
    dydt = np.empty_like(y)

    dudt = dydt[0:-2:2]
    dvdt = dydt[1:-2:2]
    dxmdt = dydt[-2]
    dxpdt = dydt[-1]

    dudt[0]    = f(u[0],    v[0],    bvec[0], gamma, n, RT, delta) + ( Du / l**2 ) * (-2.0*u[0] + 2.0*u[1]) / dx**2 - u[0] * (xpdot(u,v,xm,xp,t) - xmdot(u,v,xm,xp,t)) / l
    dudt[1:-1] = f(u[1:-1], v[1:-1], bvec[1:-1], gamma, n, RT, delta) + ( Du / l**2 ) * np.diff(u,2) / dx**2 - u[1:-1] * (xpdot(u,v,xm,xp,t) - xmdot(u,v,xm,xp,t)) / l
    dudt[-1]   = f(u[-1],   v[-1],   bvec[-1], gamma, n, RT, delta) + ( Du / l**2 ) * (-2.0*u[-1] + 2.0*u[-2]) / dx**2 - u[-1] * (xpdot(u,v,xm,xp,t) - xmdot(u,v,xm,xp,t)) / l

    dvdt[0]    = g(u[0],    v[0],    bvec[0], gamma, n, RT, delta) + ( Dv / l**2 ) * (-2.0*v[0] + 2.0*v[1]) / dx**2 - v[0] * (xpdot(u,v,xm,xp,t) - xmdot(u,v,xm,xp,t)) / l
    dvdt[1:-1] = g(u[1:-1], v[1:-1], bvec[ 1:-1], gamma, n, RT, delta) + ( Dv / l**2 ) * np.diff(v,2) / dx**2 - v[1:-1] * (xpdot(u,v,xm,xp,t) - xmdot(u,v,xm,xp,t)) / l
    dvdt[-1]   = g(u[-1],   v[-1],   bvec[-1], gamma, n, RT, delta) + ( Dv / l**2 ) * (-2.0*v[-1] + 2.0*v[-2]) / dx**2 - v[-1] * (xpdot(u,v,xm,xp,t) - xmdot(u,v,xm,xp,t)) / l

    dxmdt = xmdot(u,v,xm,xp,t)
    dxpdt = xpdot(u,v,xm,xp,t)

    dydt[0:-2:2] = dudt
    dydt[1:-2:2] = dvdt
    dydt[-2] = dxmdt
    dydt[-1] = dxpdt

    return dydt

L = 1
N = 1000

x = np.linspace(0, L, N)
dx = 1/N

T = 20000
M = 2000
t = np.linspace(0, T, M+1)

d1s = np.linspace(0,100,11)
# d1s = d1s[6:]
# d1s = np.array([0,30])
us = []
vs = []
xms = []
xps = []
ls = []


for i in range(len(d1s)):
    print(i)
    d1 = d1s[i]

    b0, b1, gamma, n, RT, d0 = 2.5, 1, 5, 6, 2, 2
    Du = 0.01
    Dv = 10

    #b  = 0.1 steadystate
    u0ic = 0.05645
    u0 = u0ic*np.ones(np.size(x))
    v0 = (RT-u0ic)*np.ones(np.size(x))

    y0 = np.zeros(2*N+2)
    xm0 = 0
    xp0 = 1
    y0[0:-2:2] = u0
    y0[1:-2:2] = v0
    y0[-2] = xm0
    y0[-1] = xp0

    sol = odeint(rdPDE, y0, t, args=(b0, b1, gamma, n, RT, d0, d1, Du, Dv, dx))
    # sol = solve_ivp(lambda t,y: rdPDE(t, y, b0, b1, gamma, n, RT, d0, d1, Du, Dv, dx), (0,T), y0, method='Radau', rtol=1e-8, atol=1e-8, t_eval=np.linspace(0,T,500))

    # t = sol.t
    # y = sol.y
    u = sol[:,0:-2:2]
    v = sol[:,1:-2:2]
    xm = sol[:,-2]
    xp = sol[:,-1]
    # u = y[0:-2:2,:].T
    # v = y[1:-2:2,:].T
    # xm = y[-2,:].T
    # xp = y[-1,:].T
    l = xp - xm

    us.append(u)
    vs.append(v)
    xms.append(xm)
    xps.append(xp)
    ls.append(l)

    Xgrid, Tgrid = np.meshgrid(x,t)
    Xgrid = np.empty_like(Xgrid)
    for i in range(len(t)):
        Xgrid[i,:] = x*l[i]+xm[i]

    fig, ax1 = plt.subplots(1,figsize=(2.6,2.6))
    pmesh = plt.pcolormesh(Xgrid,Tgrid,u,cmap=cm.hot,vmin=0,vmax=2)
    cbar = fig.colorbar(pmesh,ax=ax1)
    xmline = plt.plot(xm,t,linewidth=1.5,color='k')
    xpline = plt.plot(xp,t,linewidth=1.5,color='k')
    # plt.axhline(y=20,linestyle='--',linewidth=2,color='w')
    cbar.outline.set_linewidth(1.5)
    cbar.ax.tick_params(width=1.5)
    ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)

    # ax1.set_xlabel(r'$x$')
    # ax1.set_ylabel(r'$t$')
    # ax1.set_title(r'$R(x,t)$')
    ax1.set_facecolor((0.7,0.7,0.7))
    ax1.spines["left"].set_linewidth(1.5)
    ax1.spines["top"].set_linewidth(1.5)
    ax1.spines["right"].set_linewidth(1.5)
    ax1.spines["bottom"].set_linewidth(1.5)

    ax1.set_xlim(0,9)
    # ax1.set_xticks([0,1,2,3,4])
    # ax1.set_xticklabels(['0','','2','','4'])

    xcoord = ax1.get_xlim()
    ycoord = ax1.get_ylim()
    xx = np.linspace(xcoord[0],9)
    yy = np.linspace(ycoord[0],ycoord[1])
    X,Y=np.meshgrid(xx,yy)
    ax1.pcolormesh(X,Y,b0+b1*X,cmap=cm.binary,vmin=b0,vmax=b0+b1*9,zorder=0)

    plt.tight_layout()
    plt.savefig('bGrad_b0={}_b1={}_d1={}.png'.format(b0,b1,d1),dpi=600)
    plt.close()
    # plt.savefig('Fig14_moving.eps')
    # plt.show()

    fig,ax = plt.subplots(1,figsize=(3,3))
    plt.scatter(b0+b1*xm,d0 + d1*(l-1))
    plt.scatter(b0+b1*xp,d0 + d1*(l-1))
    ax.set_ylim([0,10])
    # ax.set_xlim([0,6])
    ax.grid()
    plt.savefig('bGrad_b0={}_b1={}_d1={}_bdplane.png'.format(b0,b1,d1),dpi=600)
    plt.close()


np.savez('gradient_data.npz', us=us,vs=vs,xms=xms,xps=xps,ls=ls,x=x,t=t,d1s=d1s)
