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
    dvdt[1:-1] = g(u[1:-1], v[1:-1], bvec[ 1:-1], gamma, n, RT, delta) + ( Dv / l**2 ) * np.diff(v,2) / dx**2 - v[1:-1] * (xpdot(u,v,xm,xp,t) - xmdot(u,v,xm,xp,t)) / l
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
N = 1000
x = np.linspace(0, L, N)
dx = 1/N
T = 250
M = 1000
t = np.linspace(-5, T, M)

# d1s = np.linspace(0,150,16)
d1s = [0.0]
for i in range(len(d1s)):
    d1 = d1s[i]

    b0, b1, gamma, n, RT, d0 = 4, 0, 5, 6, 2, 3
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

    # scipy.io.savemat('data.mat',dict(t=t,l=l,u=u))

    # fig = plt.figure("fig1",figsize=(5.2,2.6))
    # ax1 = plt.subplot(121)
    # pmesh = plt.pcolormesh(x,t,u,cmap=cm.hot)
    # cbar = fig.colorbar(pmesh,ax=ax1)
    # # plt.axhline(y=20,linestyle='--',linewidth=2,color='w')
    # cbar.outline.set_linewidth(1.5)
    # cbar.ax.tick_params(width=1.5)
    # ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
    # ax1.set_xlabel(r'$\bar x$')
    # ax1.set_ylabel(r'$t$')
    # ax1.set_title(r'$R(\bar x,t)$')
    # ax1.spines["left"].set_linewidth(1.5)
    # ax1.spines["top"].set_linewidth(1.5)
    # ax1.spines["right"].set_linewidth(1.5)
    # ax1.spines["bottom"].set_linewidth(1.5)
    #
    # ax2 = plt.subplot(122)
    # pmesh =plt.pcolormesh(x,t,v,cmap=cm.hot)
    # ax2.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10, width=1.5)
    # cbar = fig.colorbar(pmesh,ax=ax2)
    # # plt.axhline(y=20,linestyle='--',linewidth=2,color='w')
    # cbar.outline.set_linewidth(1.5)
    # cbar.ax.tick_params(width=1.5)
    # ax2.set_xlabel(r'$\bar x$')
    # ax2.set_ylabel(r'$t$')
    # ax2.set_title(r'$R_i(\bar x,t)$')
    # ax2.spines["left"].set_linewidth(1.5)
    # ax2.spines["top"].set_linewidth(1.5)
    # ax2.spines["right"].set_linewidth(1.5)
    # ax2.spines["bottom"].set_linewidth(1.5)
    #
    # plt.tight_layout()
    # # plt.savefig('Fig14_fixed.png',dpi=1200)

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
    ax1.set_xlim(np.amin(xm)-0.1,np.amax(xp)+0.1)
    ax1.set_facecolor((0.7,0.7,0.7))
    ax1.spines["left"].set_linewidth(1.5)
    ax1.spines["top"].set_linewidth(1.5)
    ax1.spines["right"].set_linewidth(1.5)
    ax1.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('4b_fixed_d1={}.tiff'.format(d1),dpi=1200)
    # plt.savefig('Fig14_moving.eps')
    # plt.show()


# #%%
# fig,ax = plt.subplots(1,figsize=(4,3))
# # for i in range(np.mod(len(t),10)
#
# for i in range(int(np.floor(len(t)))):
#     plt.plot(x,u[i,:])
#
# plt.plot(x,u[-1,:],lw=4)
# print(np.amax(u[-1,:]) - np.amin(u[-1,:]))
#
# plt.show()
# #%%
# #
#
# fig,ax = plt.subplots(1,figsize=(3,3))
# plt.scatter(b0+b1*xm,d0 + d1*(l-1))
# plt.scatter(b0+b1*xp,d0 + d1*(l-1))
# ax.set_ylim([0,10])
# # ax.set_xlim([0,6])
# ax.grid()
# plt.savefig('length_3b.tif')
# # plt.plot(t,delta)
# #%%
#
# fig,ax1 = plt.subplots(1,figsize=(4,3))
# # plt.plot(t,l,label='$L$')
# # plt.figure()
# # plt.plot(t,d0 + d1*(l-1),label='$\delta$')
# # plt.figure()
# color = 'k'
# ax1.plot(t,RT/l, label='$R_T/L$', color=color)
# ax1.set_ylabel('$R_T/L$')
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:green'
# ax2.set_ylabel('$\delta(T)$', color=color)  # we already handled the x-label with ax1
# ax2.plot(t, d0 + d1*(l-1), color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# ax1.set_xlabel('Time')
# # ax1.legend()
# plt.tight_layout()
# plt.savefig('RTdelta.png',dpi=1200)
# plt.show()
#
# # %%
# # check mass conservation
# print(np.sum(dx*l[0]*(u[0,:] + v[0,:])))
# print(np.sum(dx*l[-1]*(u[-1,:] + v[-1,:])))
# mass = []
# for i in range(len(t)):
#     mass.append(np.sum(dx*l[i]*(u[i,:] + v[i,:])))
#
# fig = plt.figure("fig2",figsize=(4,3))
# ax1 = plt.subplot(111)
# plt.plot(t,mass,linewidth=2)
# ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
# ax1.set_xlabel(r'$t$')
# ax1.set_ylabel(r'$\int_\Omega R + R_i \, dx$')
# ax1.spines["left"].set_linewidth(1.5)
# ax1.spines["top"].set_linewidth(1.5)
# ax1.spines["right"].set_linewidth(1.5)
# ax1.spines["bottom"].set_linewidth(1.5)
# ax1.set_xlim(0,T)
# ax1.grid(linewidth=1.5)
# plt.tight_layout()
# # plt.savefig('Fig14mass.png',dpi=1200)
# plt.show()
#
# # #%%
# # # animated plot
#
movieon = 1
if movieon == 1:
    import matplotlib.animation as animation
    fig = plt.figure(figsize=(4,3))
    ax = plt.subplot(111)
    ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel('Activity')
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.set_xlim(np.min(Xgrid),np.max(Xgrid))
    ax.set_ylim(0,np.max(u)+0.5)
    ax.grid(linewidth=1.5)
    # title = plt.title(r'$b$=%1.2f, $\delta$=%1.2f' %(b, delta))
    line_u, = ax.plot(Xgrid[0,:],u[0,:],linewidth=4,color=(221/255,170/255,51/255),label=r'$R$')
    # line_v, = ax.plot(Xgrid[0,:],v[0,:],'--',linewidth=2,color=(0/255,68/255,136/255),label=r'$R_i$')
    plt.legend(loc=2)
    vertline = ax.axvline(Xgrid[0,-1],ls='-',color='k')
    vertline2 = ax.axvline(Xgrid[0,0],ls='-',color='k')
    ax.set_xlim(np.amin(xm)-0.1,np.amax(xp)+0.1)
    plt.tight_layout()

    def animate(i):
        # title.set_text(r'$b$=%1.2f, $\delta$=%1.2f' %(bvals[i], deltavals[i]))
        line_u.set_xdata(Xgrid[i,:])
        # line_v.set_xdata(Xgrid[i,:])
        line_u.set_ydata(u[i,:])
        # line_v.set_ydata(v[i,:])
        vertline.set_xdata(Xgrid[i,-1])
        vertline2.set_xdata(Xgrid[i,0])
        return line_u, vertline, vertline2

    ani = animation.FuncAnimation(fig,animate,frames=len(t))
    ani.save("Movie_4b.mp4",fps=30,dpi=300)
