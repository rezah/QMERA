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


R=np.loadtxt("meraX8.txt")
xX8=R[:,0]
yX8=R[:,1]



R=np.loadtxt("meraMX8.txt")
xMX8=R[:,0]
yMX8=R[:,1]


R=np.loadtxt("meraX16.txt")
xX16=R[:,0]
yX16=R[:,1]



R=np.loadtxt("meraMX16.txt")
xMX16=R[:,0]
yMX16=R[:,1]


R=np.loadtxt("meraX12.txt")
xX12=R[:,0]
yX12=R[:,1]
plt.plot( xX8, yX8,'--',lw=3,markersize=10, color = '#f57900', label='TTN, $\chi=8$')
plt.plot( xX12, yX12,'-.',lw=3,markersize=10, color = '#cf729d', label='TTN, $\chi=12$')
plt.plot( xX16, yX16,'-.',lw=3,markersize=10, color = '#e30b69', label='TTN, $\chi=16$')
plt.plot( xMX8, yMX8,'-.',lw=3,markersize=10, color = '#5c3566', label='MERA, $\chi=8$')
plt.plot( xMX16, yMX16,'-.',lw=3,markersize=10, color = '#0e9c38', label='MERA, $\chi=16$')


#plt.title('qmps')
plt.ylabel(r'E',fontsize=21)
plt.xlabel(r'iterations',fontsize=18)
plt.axhline(-50.12,color='black', label='PEPS, D=4')


plt.xlim([20,1500])
plt.ylim([-51.0, -42])

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="upper right", prop={'size': 18})




plt.grid(True)
plt.savefig('qmpsB-plot.pdf')
plt.clf()
