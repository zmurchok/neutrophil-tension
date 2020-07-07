import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.rc("text", usetex=False)
plt.rc("font", **{'family':'sans-serif', 'sans-serif':['Helvetica'], 'size':10})

def f(u, v, b, gamma, n, RT, delta):
    return (b+gamma*u**n/(1+u**n))*v - delta*u

def g(u, v, b, gamma, n, RT, delta):
    return -f(u,v,b, gamma, n, RT, delta)

def rdPDE(t, y, b, gamma, n, RT, delta, Du, Dv, dx):
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
T = 100
# M = 1000
# print(M)
# t = np.linspace(0, T, M)


b, gamma, n, RT, delta = 0.1, 5, 6, 2, 3
Du = 0.01
Dv = 10

# print(dx**2/2*Dv)
# dt = T/M
# print(dt)

width = 0.1
height = 5
u0 = np.zeros(N)
u0[x > 1-width] = height
# u0ic = 0.05645
# u0 = u0ic*np.ones(np.size(x))
v0 = (RT-width*height)*np.ones(np.size(x))
y0 = np.zeros(2*N)
y0[::2] = u0
y0[1::2] = v0

# NON POLAR LOW RAC
b = 0.1
delta = 7.5
sol = solve_ivp(lambda t,y: rdPDE(t, y, b, gamma, n, RT, delta, Du, Dv, dx), [0,T], y0, method='LSODA', lband=2, uband=2, rtol=1e-12, atol=1e-12)

t = sol.t
y = sol.y

u = y[0::2,:].T
v = y[1::2,:].T
#
# plt.figure()
# plt.pcolormesh(u)
# plt.colorbar()

fig = plt.figure("fig1",figsize=(4,2.6))
ax = plt.subplot(221)
ax.set_xlim(0,1)
ax.set_ylim(-0.1,2.1)
ax.grid(linewidth=1.5)
# line_ic, = ax.plot(x,u[0,:],':',linewidth=4,color=(221/255,170/255,51/255),label=r'IC')
line_ss, = ax.plot(x,u[-1,:],linewidth=4,color=(221/255,170/255,51/255),label=r'Steady-state')
ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
ax.spines["left"].set_linewidth(1.5)
ax.spines["top"].set_linewidth(1.5)
ax.spines["right"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

# POLARIZABLE
b = 0.1
delta = 3
sol = solve_ivp(lambda t,y: rdPDE(t, y, b, gamma, n, RT, delta, Du, Dv, dx), [0,T], y0, method='LSODA', lband=2, uband=2, rtol=1e-12, atol=1e-12)

t = sol.t
y = sol.y

u = y[0::2,:].T
v = y[1::2,:].T

ax = plt.subplot(223)
ax.set_xlim(0,1)
ax.set_ylim(-0.1,2.1)
ax.grid(linewidth=1.5)
# line_ic, = ax.plot(x,u[0,:],':',linewidth=4,color=(221/255,170/255,51/255),label=r'IC')
line_ss, = ax.plot(x,u[-1,:],linewidth=4,color=(221/255,170/255,51/255),label=r'Steady-state')
ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
ax.spines["left"].set_linewidth(1.5)
ax.spines["top"].set_linewidth(1.5)
ax.spines["right"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

#TURING REGIME
b = 4.5
delta = 7.5

width = 0.1
height = 2
u0ic = 1
u0 = u0ic*np.ones(np.size(x))+0.1*np.sin(4*np.pi*x)
v0 = (RT-1)*np.ones(np.size(x))
y0 = np.zeros(2*N)
y0[::2] = u0
y0[1::2] = v0

sol = solve_ivp(lambda t,y: rdPDE(t, y, b, gamma, n, RT, delta, Du, Dv, dx), [0,T], y0, method='LSODA', lband=2, uband=2, rtol=1e-12, atol=1e-12)

t = sol.t
y = sol.y

u = y[0::2,:].T
v = y[1::2,:].T

ax = plt.subplot(222)
ax.set_xlim(0,1)
ax.set_ylim(-0.1,2.1)
ax.grid(linewidth=1.5)
# line_ic, = ax.plot(x,u[0,:],':',linewidth=4,color=(221/255,170/255,51/255),label=r'IC')
line_ss, = ax.plot(x,u[-1,:],linewidth=4,color=(221/255,170/255,51/255),label=r'Steady-state')
ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
ax.spines["left"].set_linewidth(1.5)
ax.spines["top"].set_linewidth(1.5)
ax.spines["right"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

#HIGH RAC, NON POLAR
b = 4.5
delta = 3
width = 0.1
height = 5
u0 = np.zeros(N)
u0[x > 1-width] = height
# u0ic = 0.05645
# u0 = u0ic*np.ones(np.size(x))
v0 = (RT-width*height)*np.ones(np.size(x))
y0 = np.zeros(2*N)
y0[::2] = u0
y0[1::2] = v0
sol = solve_ivp(lambda t,y: rdPDE(t, y, b, gamma, n, RT, delta, Du, Dv, dx), [0,T], y0, method='LSODA', lband=2, uband=2, rtol=1e-12, atol=1e-12)

t = sol.t
y = sol.y

u = y[0::2,:].T
v = y[1::2,:].T

ax = plt.subplot(224)
ax.set_xlim(0,1)
ax.set_ylim(-0.1,2.1)
ax.grid(linewidth=1.5)
# line_ic, = ax.plot(x,u[0,:],':',linewidth=4,color=(221/255,170/255,51/255),label=r'IC')
line_ss, = ax.plot(x,u[-1,:],linewidth=4,color=(221/255,170/255,51/255),label=r'Steady-state')
ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
ax.spines["left"].set_linewidth(1.5)
ax.spines["top"].set_linewidth(1.5)
ax.spines["right"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('Fig2b.tiff',dpi=1200)
plt.show()


# %%
# check mass conservation
print(np.sum(dx*(u[0,:] + v[0,:])))
print(np.sum(dx*(u[-1,:] + v[-1,:])))
mass = []
for i in range(len(t)):
    mass.append(np.sum(dx*(u[i,:] + v[i,:])))

fig = plt.figure("fig2",figsize=(4,3))
ax1 = plt.subplot(111)
plt.plot(t,mass,linewidth=4)
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
# plt.savefig('Fig1.png',dpi=600)
plt.show()
