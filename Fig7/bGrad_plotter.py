import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

plt.rc("text", usetex=False)
plt.rc("font", **{'family':'sans-serif', 'sans-serif':['Arial'], 'size':10})


npzfile = np.load('gradient_data.npz')
d1s = npzfile.f.d1s
x = npzfile.f.x
t = npzfile.f.t
us = npzfile.f.us
vs = npzfile.f.vs
xms = npzfile.f.xms
xps = npzfile.f.xps
ls = npzfile.f.ls
# d1s = d1s[:5]
T = 20000
M = 2000
t = np.linspace(0, T, M+1)

#%%
# sns.set_palette('Blues',len(d1s))
# plt.rc("font", **{'family':'sans-serif', 'sans-serif':['Arial'], 'size':10})

fig1, ax1 = plt.subplots(1,figsize=(4/(4/3),3/(4/3)))
for i in range(len(d1s)):
    R = us[i]
    # R[:,0]
    #R[:,0] = x_0
    # R[0,:] = R(x,t = 0)
    q75 = np.empty_like(R[:,0])
    q25 = np.empty_like(R[:,0])
    for j in range(len(R[:,0])):
        q75[j] = np.percentile(R[j,:],75)
        q25[j] = np.percentile(R[j,:],25)
    # p = ax1.plot((xms[i]+xps[i])/2,t,label='{}'.format(d1s[i]),lw=3)
    p = ax1.scatter((xms[i]+xps[i])/2,t,1,c=(q75-q25),cmap=cm.hot,vmin=0,vmax=1)

ax1.spines["left"].set_linewidth(1.5)
ax1.spines["top"].set_linewidth(1.5)
ax1.spines["right"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)
ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
# ax1.set_xticks([0,4,8])
ax1.set_xlim([0,9])
# ax1.set_yticks([0,5000,10000,15000,20000])
# ax1.set_yticklabels(['0','','','','20000'])
ax1.set_ylabel('Time')
ax1.set_xlabel('Position of cell centroid')
# ax1.set_xticklabels
ax1.set_ylim(0,20000)
# ax1.legend(loc=2,fontsize=8,title='$\delta_1$',ncol=2)
cbar = plt.colorbar(p)#,shrink=0.5)
cbar.outline.set_linewidth(1.5)
cbar.ax.tick_params(width=1.5)
cbar.ax.set_ylabel('IQR')
plt.tight_layout()
# plt.savefig('distance.tiff',dpi=600)
plt.savefig('distance.eps',format='eps')

#%%

fig2, ax2 = plt.subplots(1,figsize=(2.6,2.6))
for i in range(len(d1s)):
    R = us[i]
    # R[:,0]
    #R[:,0] = x_0
    # R[0,:] = R(x,t = 0)
    q75 = np.empty_like(R[:,0])
    q25 = np.empty_like(R[:,0])
    for j in range(len(R[:,0])):
        q75[j] = np.percentile(R[j,:],75)
        q25[j] = np.percentile(R[j,:],25)
    ax2.plot(t,q75-q25,lw=2)
    # ax2.plot(t,(np.amax(us[i],1)-np.amin(us[i],1)),label='{}'.format(d1s[i]),lw=2)
ax2.spines["left"].set_linewidth(1.5)
ax2.spines["top"].set_linewidth(1.5)
ax2.spines["right"].set_linewidth(1.5)
ax2.spines["bottom"].set_linewidth(1.5)
ax2.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)
# ax2.set_xlabel('Time')
ax2.set_xticks([0,20000])
# ax2.set_ylabel('max - min Rac')
# ax2.set_xlim(0,20000)
# ax2.set_ylim(0.1,0.8)
plt.tight_layout()
plt.savefig('polarity.tiff',dpi=600)
plt.show()


#%%
plt.rc("font", **{'family':'sans-serif', 'sans-serif':['Helvetica'], 'size':10})

u = us[0]
v = vs[0]
l = ls[0]
xp = xps[0]
xm = xms[0]
# t = np.linspace(0,20000,2001)
Xgrid, Tgrid = np.meshgrid(x,t)
Xgrid = np.empty_like(Xgrid)
for i in range(len(t)):
    Xgrid[i,:] = x*l[i]+xm[i]

fig, ax1 = plt.subplots(1,figsize=(2,2))
pmesh = plt.pcolormesh(Xgrid,Tgrid,u,cmap=cm.hot,vmin=0,vmax=2,zorder=1)
xmline = plt.plot(xm,t,linewidth=1.5,color='k',zorder=1)
xpline = plt.plot(xp,t,linewidth=1.5,color='k',zorder=1)

u = us[3]
v = vs[3]
l = ls[3]
xp = xps[3]
xm = xms[3]
Xgrid, Tgrid = np.meshgrid(x,t)
Xgrid = np.empty_like(Xgrid)
for i in range(len(t)):
    Xgrid[i,:] = x*l[i]+xm[i]
pmesh = plt.pcolormesh(Xgrid,Tgrid,u,cmap=cm.hot,vmin=0,vmax=2,zorder=2)
xmline = plt.plot(xm,t,linewidth=1.5,color='k',zorder=2)
xpline = plt.plot(xp,t,linewidth=1.5,color='k',zorder=2)

u = us[-1]
v = vs[-1]
l = ls[-1]
xp = xps[-1]
xm = xms[-1]
Xgrid, Tgrid = np.meshgrid(x,t)
Xgrid = np.empty_like(Xgrid)
for i in range(len(t)):
    Xgrid[i,:] = x*l[i]+xm[i]
pmesh = plt.pcolormesh(Xgrid,Tgrid,u,cmap=cm.hot,vmin=0,vmax=2,zorder=3)
xmline = plt.plot(xm,t,linewidth=1.5,color='k',zorder=3)
xpline = plt.plot(xp,t,linewidth=1.5,color='k',zorder=3)

# cbar = fig.colorbar(pmesh,ax=ax1)
# plt.axhline(y=20,linestyle='--',linewidth=2,color='w')
# cbar.outline.set_linewidth(1.5)
# cbar.ax.tick_params(width=1.5)
ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10 , width=1.5)

# ax1.set_xlabel(r'$x$')
# ax1.set_ylabel(r'$t$')
# ax1.set_title(r'$R(x,t)$')
ax1.set_facecolor((0.7,0.7,0.7))
ax1.spines["left"].set_linewidth(1.5)
ax1.spines["top"].set_linewidth(1.5)
ax1.spines["right"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)

# ax1.set_xticks([0,1,2,3,4])
# ax1.set_xticklabels(['0','','2','','4'])
b0 = 2.5
b1 = 1
xcoord = ax1.get_xlim()
ycoord = ax1.get_ylim()
xx = np.linspace(xcoord[0],10)
yy = np.linspace(ycoord[0],ycoord[1])
X,Y=np.meshgrid(xx,yy)
ax1.pcolormesh(X,Y,b0+b1*X,cmap=cm.binary,vmin=b0,vmax=b0+b1*10,zorder=0)
# ax1.set_xticks([0,4,8])
ax1.set_xlim([0,9])
ax1.set_yticks([0,10000,20000])
# ax1.set_yticklabels(['0','','','','20000'])
ax1.set_yticklabels([])
ax1.set_xticks([0,3,6,9])
# ax1.set_xticklabels()
# ax1.set_xticklabels([])
# ax1.set_ylabel('Time')
# ax1.set_xlabel('$x$')

plt.tight_layout()
plt.savefig('test.tiff',dpi=600)
# plt.close()

plt.show()
d1s[0]
d1s[3]
d1s[-1]
#%%
plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial'],'size':10})

fig, axs = plt.subplots(2,3,figsize=(6,4),constrained_layout=True)

ax1 = axs[0,0]
ax2 = axs[0,1]
ax3 = axs[0,2]
ax4 = axs[1,0]
ax5 = axs[1,1]
ax6 = axs[1,2]
axes = [ax1,ax2,ax3,ax4,ax5,ax6]
v_max = 2
times = [10,50,100,250,500,1000]#,1500,2000]
# idx = [0,199,399,499,749,999]
# times = [0,5,10,15,20,25]
# t = 25
# data = Rac_100[:,t]
Rac_00 = us[0]
np.shape(Rac_00)
Rac_30 = us[3]
Rac_100 = us[-1]
np.shape(Rac_100)
x = np.linspace(0,2,200)
from scipy import stats
import statsmodels.api as sm
from scipy.signal import find_peaks
bandwidth=0.05
for j in range(len(times)):
    time = t[times[j]]
    ax = axes[j]

    # ax.set_yticks([-2,0])
    # ax.set_xlim([-0.75,12])
    # ax.set_xticks([0,5,10])
    # ax.set_xticklabels([])
    # ax.set_ylim([-3,1])
    # ax.set_yticklabels([])
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(direction='in', right=True, top=True, width=1.5)
    bins = np.linspace(0,2,100)
    ax.hist(Rac_00[times[j],:],bins=bins,alpha=0.5,density=1,label='0')
    ax.hist(Rac_30[times[j],:],bins=bins,alpha=0.5,density=1,label='30')
    ax.hist(Rac_100[times[j],:],bins=bins,alpha=0.5,density=1,label='100')
    kde_00 = sm.nonparametric.KDEUnivariate(Rac_00[times[j],:])
    kde_00.fit(bw=bandwidth)
    kde_30 = sm.nonparametric.KDEUnivariate(Rac_30[times[j],:])
    kde_30.fit(bw=bandwidth)
    kde_100 = sm.nonparametric.KDEUnivariate(Rac_100[times[j],:])
    kde_100.fit(bw=bandwidth)
    ax.plot(kde_00.support,kde_00.density,c='tab:blue')
    ax.plot(kde_30.support,kde_30.density,c='tab:orange')
    ax.plot(kde_100.support,kde_100.density,c='tab:green')
    # peaks_00, _ = find_peaks(kde_00.density)
    # peaks_100, _ = find_peaks(kde_100.density)
    # ax.plot(kde_00.support[peaks_00],kde_00.density[peaks_00],'x',c='tab:blue')
    # ax.plot(kde_100.support[peaks_100],kde_100.density[peaks_100],'x',c='tab:orange')
    # labels=['0','30']
    # ax.boxplot([Rac_00[:,j],Rac_100[:,j]],labels=labels)
    # cbar = plt.colorbar(p,ax=ax)
    ax.set_xlim([0,2])
    ax.set_title('t = {:.0f}'.format(time),fontsize=10)

axes[4].set_xlabel('$R$')
axes[0].set_ylabel('Probability Density')
axes[3].set_ylabel('Probability Density')
axes[-1].legend(title='$\delta_1$')
# fig.savefig('hist.png',format='png',dpi=600)
fig.savefig('hist.pdf',format='pdf')
plt.show()
