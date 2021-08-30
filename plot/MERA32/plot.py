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



R=np.loadtxt("meraF.txt")
x1=R[:,0]
y1=R[:,1]

R=np.loadtxt("meraFF.txt")
x2=R[:,0]
y2=R[:,1]


R=np.loadtxt("meraFFF.txt")
x3=R[:,0]
y3=R[:,1]

y=list(y1)+list(y2)+list(y3)

#y=sort_low(y)


#R=np.loadtxt("meraX12F.txt")
#xMX12=R[:,0]
#yMX12=R[:,1]


#R=np.loadtxt("meraX18F.txt")
#xMX18=R[:,0]
#yMX18=R[:,1]


#R=np.loadtxt("meraX22F.txt")
#xMX22=R[:,0]
#yMX22=R[:,1]


#R=np.loadtxt("meraX28F.txt")
#xMX28=R[:,0]
#yMX28=R[:,1]


fig=plt.figure(figsize=(7,7))

plt.plot( y,'--',lw=3,markersize=10, color = '#f57900', label='MERA, $\chi=8$')
#plt.plot(  yMX12,'-.',lw=3,markersize=10, color = '#cf729d', label='MERA, $\chi=12$')
#plt.plot(  yMX18,'-.',lw=3,markersize=10, color = '#0e9c38', label='MERA, $\chi=18$')
#plt.plot(  yMX22,'-.',lw=3,markersize=10, color = '#400e9c', label='MERA, $\chi=22$')
#plt.plot(  yMX28,'-.',lw=3,markersize=10, color = '#f12626', label='MERA, $\chi=28$')


plt.xscale('log')


#plt.title('qmps')
plt.xlabel(r'iterations',fontsize=18)
plt.ylabel(r'E',fontsize=21)
plt.axhline(-160.012834,color='black', label='PEPS, D=4')


#plt.xlim([20,1000])
#plt.ylim([-40, -34])

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="upper right", prop={'size': 18})




plt.grid(True)
plt.savefig('qmpsB-plot.pdf')
plt.clf()
