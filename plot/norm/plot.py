from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import linalg as LA

mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.minor.width'] = 1




def sort_low(error):
 val_min=1.e8
 error_f=[]
 for i in range(len(error)-1):
   if error[i]>=error[i+1] and  error[i]<=val_min :
    error_f.append(error[i])
    val_min=error[i+1]*1.0
    pass

 return error_f



R=np.loadtxt("mgs.txt")
x=R[:,0]
y=R[:,1]


#y=sort_low(y)


R=np.loadtxt("svd.txt")
xMX12=R[:,0]
yMX12=R[:,1]


R=np.loadtxt("exp.txt")
xMX18=R[:,0]
yMX18=R[:,1]


R=np.loadtxt("qr.txt")
xMX22=R[:,0]
yMX22=R[:,1]


# R=np.loadtxt("meraX28F.txt")
# xMX28=R[:,0]
# yMX28=R[:,1]


fig=plt.figure(figsize=(7,7))

plt.plot( y,'--',lw=3,markersize=10, color = '#f57900', label='mgs')
plt.plot(  yMX12,'-.',lw=3,markersize=10, color = '#cf729d', label='svd')
plt.plot(  yMX18,'-.',lw=3,markersize=10, color = '#0e9c38', label='exp')
plt.plot(  yMX22,'-.',lw=3,markersize=10, color = '#400e9c', label='qr')
#plt.plot(  yMX28,'-.',lw=3,markersize=10, color = '#f12626', label='MERA, $\chi=28$')


plt.xscale('log')


#plt.title('qmps')
plt.xlabel(r'iterations',fontsize=18)
plt.ylabel(r'E',fontsize=21)
#plt.axhline(-39.212834,color='black', label='PEPS, D=4')


plt.xlim([700,1200])
plt.ylim([-154.5, -153])

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="upper right", prop={'size': 18})




plt.grid(True)
plt.savefig('qmpsB-plot_f.pdf')
plt.clf()
