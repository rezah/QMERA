from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import linalg as LA

#mpl.rcParams['xtick.major.size'] = 10
#mpl.rcParams['xtick.major.width'] = 1
#mpl.rcParams['xtick.minor.size'] = 5
#mpl.rcParams['xtick.minor.width'] = 1
#mpl.rcParams['ytick.major.size'] = 10
#mpl.rcParams['ytick.major.width'] = 1
#mpl.rcParams['ytick.minor.size'] = 5
#mpl.rcParams['ytick.minor.width'] = 1




def sort_low(error):
 val_min=1.e8
 error_f=[]
 for i in range(len(error)-1):
   if error[i]>=error[i+1] and  error[i]<=val_min :
    error_f.append(error[i])
    val_min=error[i+1]*1.0
    pass

 return error_f



#R=np.loadtxt("meraF.txt")
#x1=R[:,0]
#y1=R[:,1]

#R=np.loadtxt("meraFF.txt")
#x2=R[:,0]
#y2=R[:,1]


R=np.loadtxt("swidth.txt")
xs=R[:,0]
ys=R[:,1]
zs=R[:,2]
es=R[:,3]
gs=R[:,4]



R=np.loadtxt("fwidth.txt")
xf=R[:,0]
yf=R[:,1]
zf=R[:,2]
ef=R[:,3]
gf=R[:,4]



#fig=plt.figure(figsize=(7,7))

#plt.plot( y,'--',lw=3,markersize=10, color = '#f57900', label='MERA, $\chi=8$')
#plt.plot(  yMX12,'-.',lw=3,markersize=10, color = '#cf729d', label=r'$ \tau=2, \chi=22$')
#plt.plot(  yMX18,'-.',lw=3,markersize=10, color = '#0e9c38', label=r'MERA, $\chi=30$')
plt.plot(  es,gs,'o',lw=3,markersize=8, color = '#400e9c', label=r'sMERA, $\chi=210$')
plt.plot(  ef,gf,'+',lw=3,markersize=8, color = '#f12626', label=r"$fMERA, \chi=22$")


#plt.xscale('log')
plt.yscale('log')


#plt.title('qmps')
plt.ylabel(r'$\delta E$',fontsize=14)
plt.xlabel(r'peakmemory',fontsize=14)
#plt.axhline(-166.444*(1./(6.**3.)),color='#400e9c', label=r'MPS, $\chi=700$')


#plt.xlim([20,12000])
#plt.ylim([-0.7735, -0.764])

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc="upper right", prop={'size': 13})
plt.ylim([5.e-3, 1.e-1])




plt.grid(True)
plt.savefig('plotpeak.pdf')
plt.clf()
