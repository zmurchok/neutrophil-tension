import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

plt.rc("text", usetex=False)
plt.rc("font", **{'family':'sans-serif', 'sans-serif':['Helvetica'], 'size':10})

t = []
A = []
M = []
P = []

t,A,M,P = np.genfromtxt('SIFig2.csv',skip_header=1,unpack=True,delimiter=',')


# fig = plt.figure(figsize=(2.6,2.6))
# ax1 = plt.subplot(111)
# ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10, width=1.5)
# plt.plot(t,A)
# ax1.set_xlabel(r'$t$')
# ax1.set_ylabel(r'$A$')
# ax1.spines["left"].set_linewidth(1.5)
# ax1.spines["top"].set_linewidth(1.5)
# ax1.spines["right"].set_linewidth(1.5)
# ax1.spines["bottom"].set_linewidth(1.5)
# ax1.grid(linewidth=1.5)
# plt.tight_layout()
# plt.savefig('Fig6_Area.tiff',dpi=600)
# plt.savefig('Fig6_Area.eps',dpi=600)

#plt.show()

fig = plt.figure(figsize=(3,3))
ax1 = plt.subplot(111)
ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10, width=1.5)
plt.plot(t,M)
ax1.set_xlabel(r'Time Step')
ax1.set_ylabel(r'Total Rac')
ax1.spines["left"].set_linewidth(1.5)
ax1.spines["top"].set_linewidth(1.5)
ax1.spines["right"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)
ax1.grid(linewidth=1.5)
plt.tight_layout()
plt.savefig('SIFig2.tiff',dpi=1200)
# plt.savefig('Fig6_Mass.eps',dpi=600)

#plt.show()
#
# fig = plt.figure(figsize=(2.6,2.6))
# ax1 = plt.subplot(111)
# ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=10, width=1.5)
# plt.plot(t,P)
# ax1.set_xlabel(r'$t$')
# ax1.set_ylabel(r'$R_{Max}-R_{Min}$')
# ax1.spines["left"].set_linewidth(1.5)
# ax1.spines["top"].set_linewidth(1.5)
# ax1.spines["right"].set_linewidth(1.5)
# ax1.spines["bottom"].set_linewidth(1.5)
# ax1.grid(linewidth=1.5)
# plt.tight_layout()
# plt.savefig('Fig6_Polarity.tiff',dpi=600)
# plt.savefig('Fig6_Polarity.eps',dpi=600)
#
# #plt.show()
