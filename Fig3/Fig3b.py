import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.rc("text", usetex=False)
plt.rc("font", **{'family':'sans-serif', 'sans-serif':['Helvetica'], 'size':10})

def f(u, v, b, gamma, n, RT, delta):
    return (b+gamma*u**n/(1+u**n))*v - delta*u

def g(u, v, b, gamma, n, RT, delta):
    return -f(u,v,b, gamma, n, RT, delta)

def rdPDE(y, t, b, gamma, n, RT, delta, Du, Dv, dx):
    """
    The ODEs are derived using the method of lines.
    https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#mol
    """
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    u = y[::2]
    v = y[1::2]

    # dydt is the return value of this function.
    dydt = np.empty_like(y)

    dudt = dydt[::2]
    dvdt = dydt[1::2]

    if (t>0) & (t<=5):
        # b = b + 4/50*(t-50)
        b = 4
    elif (t>5) & (t<=50):
        b = 4
        delta = delta+4/45*(t-5)
    elif t>50:
        b = 4
        delta = delta+4/45*(50-5)

    #this is neumann:
    dudt[0]    = f(u[0],    v[0],    b, gamma, n, RT, delta) + Du * (-2.0*u[0] + 2.0*u[1]) / dx**2
    dudt[1:-1] = f(u[1:-1], v[1:-1], b, gamma, n, RT, delta) + Du * np.diff(u,2) / dx**2
    dudt[-1]   = f(u[-1],   v[-1],   b, gamma, n, RT, delta) + Du * (-2.0*u[-1] + 2.0*u[-2]) / dx**2
    dvdt[0]    = g(u[0],    v[0],    b, gamma, n, RT, delta) + Dv * (-2.0*v[0] + 2.0*v[1]) / dx**2
    dvdt[1:-1] = g(u[1:-1], v[1:-1], b, gamma, n, RT, delta) + Dv * np.diff(v,2) / dx**2
    dvdt[-1]   = g(u[-1],   v[-1],   b, gamma, n, RT, delta) + Dv * (-2.0*v[-1] + 2.0*v[-2]) / dx**2

    dydt[::2] = dudt
    dydt[1::2] = dvdt

    return dydt

# %%
# %%time
L = 1
N = 1000
x = np.linspace(0, L, N)
dx = 1/N
T = 120
M = 1000
t = np.linspace(-5, T, M)

b, gamma, n, RT, delta = 0.1, 5, 6, 2, 3
Du = 0.01
Dv = 10

# width = 0.1
# height = 2
# u0 = np.zeros(N)
# u0[x > 1-width] = height
u0ic = 0.05645
u0 = u0ic*np.ones(np.size(x))
v0 = (RT-u0ic)*np.ones(np.size(x))
y0 = np.zeros(2*N)
y0[::2] = u0
y0[1::2] = v0

sol = odeint(rdPDE, y0, t, args=(b, gamma, n, RT, delta, Du, Dv, dx), ml=2, mu=2)

# %%

u = sol[:,::2]
v = sol[:,1::2]

fig = plt.figure("fig1",figsize=(2.6,2.6))
ax1 = plt.subplot(111)
pmesh = plt.pcolormesh(x,t,u,cmap=cm.hot,vmin=0,vmax=2)
cbar = fig.colorbar(pmesh,ax=ax1)
# plt.axhline(y=50,linestyle='--',linewidth=2,color='0.5')
# plt.axhline(y=100,linestyle='--',linewidth=2,color='0.5')
# plt.axhline(y=250,linestyle='--',linewidth=2,color='0.5')
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(width=1.5)
ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)

ax1.spines["left"].set_linewidth(1.5)
ax1.spines["top"].set_linewidth(1.5)
ax1.spines["right"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('Fig3b.tiff',dpi=1200)
plt.show()


# %%
# check mass conservation
print(np.sum(dx*(u[0,:] + v[0,:])))
print(np.sum(dx*(u[-1,:] + v[-1,:])))
mass = []
for i in range(len(t)):
    mass.append(np.sum(dx*(u[i,:] + v[i,:])))

fig = plt.figure("fig2",figsize=(2.6,2.6))
ax1 = plt.subplot(111)
plt.plot(t,mass,linewidth=2)
ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$\int_\Omega R + R_i \, dx$')
ax1.spines["left"].set_linewidth(1.5)
ax1.spines["top"].set_linewidth(1.5)
ax1.spines["right"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)
ax1.set_xlim(0,T)
ax1.grid(linewidth=1.5)
plt.tight_layout()
plt.savefig('Fig3bmass.tiff')
plt.show()


#%%
# animated plot
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
ax.set_xlim(0,1)
ax.set_ylim(0,2)
ax.grid(linewidth=1.5)
title = plt.title(r'$b$=%1.2f, $\delta$=%1.2f' %(b, delta))
line_u, = ax.plot(x,u[0,:],linewidth=4,color=(221/255,170/255,51/255),label=r'$R$')
# line_v, = ax.plot(x,v[0,:],'--',linewidth=2,color=(0/255,68/255,136/255),label=r'$R_i$')
plt.legend(loc=2)
plt.tight_layout()

bvals = np.zeros(np.shape(t))
deltavals = np.zeros(np.shape(t))
for i in range(len(t)):
    ti = t[i]
    if ti<0:
        bvals[i] = 0.1
        deltavals[i] = delta
    elif (ti>=0) & (ti<=5):
        # bvals[i] = b + 4/50*(ti-50)
        bvals[i] = 4
        deltavals[i] = delta
    elif (ti>5) & (ti<=50):
        bvals[i] = 4
        deltavals[i] = delta+4/45*(ti-5)
    elif ti>50:
        bvals[i] = 4
        deltavals[i] = delta+4/45*(50-5)
    else:
        bvals[i] = b
        deltavals[i] = delta

def animate(i):
    title.set_text(r'$t$=%1.2f, $b$=%1.2f, $\delta$=%1.2f' %(t[i],bvals[i], deltavals[i]))
    line_u.set_ydata(u[i,:])
    # line_v.set_ydata(v[i,:])
    return line_u, title #line_v, title

ani = animation.FuncAnimation(fig,animate,frames=len(t))
ani.save("Movie 3b.mp4",fps=30,dpi=300)
