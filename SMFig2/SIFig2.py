import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.io

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

    if t < 0:
        bvec = 0.1*np.ones(np.shape(np.linspace(0,1,1/dx)))
    else:
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
    dvdt[1:-1] = g(u[1:-1], v[1:-1], bvec[1:-1], gamma, n, RT, delta) + ( Dv / l**2 ) * np.diff(v,2) / dx**2 - v[1:-1] * (xpdot(u,v,xm,xp,t) - xmdot(u,v,xm,xp,t)) / l
    dvdt[-1]   = g(u[-1],   v[-1],   bvec[-1], gamma, n, RT, delta) + ( Dv / l**2 ) * (-2.0*v[-1] + 2.0*v[-2]) / dx**2 - v[-1] * (xpdot(u,v,xm,xp,t) - xmdot(u,v,xm,xp,t)) / l

    dxmdt = xmdot(u,v,xm,xp,t)
    dxpdt = xpdot(u,v,xm,xp,t)

    dydt[0:-2:2] = dudt
    dydt[1:-2:2] = dvdt
    dydt[-2] = dxmdt
    dydt[-1] = dxpdt

    return dydt

# %%
# %%time
L = 1
N = 100
x = np.linspace(0, L, N)
dx = 1/N
T = 500
M = 1000
t = np.linspace(-5, T, M)

b0, b1, gamma, n, RT, d0, d1 = 4, 0, 5, 6, 2, 3, 80
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

# %%

u = sol[:,0:-2:2]
v = sol[:,1:-2:2]
xm = sol[:,-2]
xp = sol[:,-1]
l = xp - xm

# scipy.io.savemat('data.mat',dict(t=t,l=l,u=u))

# %%
fig = plt.figure("fig1",figsize=(5.2,2.6))
ax1 = plt.subplot(121)
pmesh = plt.pcolormesh(x,t,u,cmap=cm.inferno)
cbar = fig.colorbar(pmesh,ax=ax1)
# plt.axhline(y=20,linestyle='--',linewidth=2,color='w')
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(width=1.5)
ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
ax1.set_xlabel(r'$\bar x$')
ax1.set_ylabel(r'$t$')
ax1.set_title(r'$R(\bar x,t)$')
ax1.spines["left"].set_linewidth(1.5)
ax1.spines["top"].set_linewidth(1.5)
ax1.spines["right"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)

ax2 = plt.subplot(122)
pmesh =plt.pcolormesh(x,t,v,cmap=cm.inferno)
ax2.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10, width=1.5)
cbar = fig.colorbar(pmesh,ax=ax2)
# plt.axhline(y=20,linestyle='--',linewidth=2,color='w')
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(width=1.5)
ax2.set_xlabel(r'$\bar x$')
ax2.set_ylabel(r'$t$')
ax2.set_title(r'$R_i(\bar x,t)$')
ax2.spines["left"].set_linewidth(1.5)
ax2.spines["top"].set_linewidth(1.5)
ax2.spines["right"].set_linewidth(1.5)
ax2.spines["bottom"].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('test3_fixed.tiff',dpi=600)
plt.show()

# %%
Xgrid, Tgrid = np.meshgrid(x,t)
Xgrid = np.empty_like(Xgrid)
for i in range(len(t)):
    Xgrid[i,:] = x*l[i]+xm[i]
# %%

fig = plt.figure("fig1",figsize=(2.6,2.6))
ax1 = plt.subplot(111)
pmesh = plt.pcolormesh(Xgrid,Tgrid,u,cmap=cm.inferno,vmin=0,vmax=2)
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


plt.tight_layout()
plt.savefig('test3.tiff',dpi=600)
plt.show()
#%%
fig = plt.figure(figsize=(2.6,2.6))
ax1 = plt.subplot(111)
ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
plt.plot(t,l)
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$l$')
ax1.spines["left"].set_linewidth(1.5)
ax1.spines["top"].set_linewidth(1.5)
ax1.spines["right"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)
ax1.set_xlim(0,T)
ax1.grid(linewidth=1.5)
plt.tight_layout()
plt.savefig('test3_length.tiff',dpi=600)
plt.show()
# %%
# check mass conservation
print(np.sum(dx*l[0]*(u[0,:] + v[0,:])))
print(np.sum(dx*l[-1]*(u[-1,:] + v[-1,:])))
mass = []
for i in range(len(t)):
    mass.append(np.sum(dx*l[i]*(u[i,:] + v[i,:])))

#%%

N = 200
x = np.linspace(0, L, N)
dx = 1/N
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
sol2 = odeint(rdPDE, y0, t, args=(b0, b1, gamma, n, RT, d0, d1, Du, Dv, dx))
u = sol2[:,0:-2:2]
v = sol2[:,1:-2:2]
xm = sol2[:,-2]
xp = sol2[:,-1]
l = xp - xm
mass2 = []
for i in range(len(t)):
    mass2.append(np.sum(dx*l[i]*(u[i,:] + v[i,:])))


N = 400
x = np.linspace(0, L, N)
dx = 1/N
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
sol3 = odeint(rdPDE, y0, t, args=(b0, b1, gamma, n, RT, d0, d1, Du, Dv, dx))
u = sol3[:,0:-2:2]
v = sol3[:,1:-2:2]
xm = sol3[:,-2]
xp = sol3[:,-1]
l = xp - xm
mass3 = []
for i in range(len(t)):
    mass3.append(np.sum(dx*l[i]*(u[i,:] + v[i,:])))


N = 800
x = np.linspace(0, L, N)
dx = 1/N
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
sol4 = odeint(rdPDE, y0, t, args=(b0, b1, gamma, n, RT, d0, d1, Du, Dv, dx))
u = sol4[:,0:-2:2]
v = sol4[:,1:-2:2]
xm = sol4[:,-2]
xp = sol4[:,-1]
l = xp - xm
mass4 = []
for i in range(len(t)):
    mass4.append(np.sum(dx*l[i]*(u[i,:] + v[i,:])))


N = 1600
x = np.linspace(0, L, N)
dx = 1/N
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
sol5 = odeint(rdPDE, y0, t, args=(b0, b1, gamma, n, RT, d0, d1, Du, Dv, dx))
u = sol5[:,0:-2:2]
v = sol5[:,1:-2:2]
xm = sol5[:,-2]
xp = sol5[:,-1]
l = xp - xm
mass5 = []
for i in range(len(t)):
    mass5.append(np.sum(dx*l[i]*(u[i,:] + v[i,:])))



#%%
from mpltools import annotation

N = [100,200,400,800,1600]

def error(mass):
    return np.max(np.abs(np.array(mass)-RT))

errors = [error(mass),error(mass2),error(mass3),error(mass4),error(mass5)]

fig = plt.figure("fig3",figsize=(3,3))
ax1 = plt.subplot(111)
plt.loglog(N,errors,'o-')
ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
ax1.set_xlabel(r'$N$')
ax1.set_ylabel(r'$|| e ||_\infty$')
ax1.spines["left"].set_linewidth(1.5)
ax1.spines["top"].set_linewidth(1.5)
ax1.spines["right"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)
# ax1.set_xlim(0,T)
ax1.grid(linewidth=1.5)
annotation.slope_marker((350, 0.0005), (-1,1))
# plt.legend(location='center right')
plt.tight_layout()
plt.savefig('test3_mass_error.tiff',dpi=1200)
# plt.savefig('test3_mass_error.eps')
plt.show()
