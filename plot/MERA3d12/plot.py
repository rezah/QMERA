from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import linalg as LA

# mpl.rcParams['xtick.major.size'] = 10
# mpl.rcParams['xtick.major.width'] = 1
# mpl.rcParams['xtick.minor.size'] = 5
# mpl.rcParams['xtick.minor.width'] = 1
# mpl.rcParams['ytick.major.size'] = 10
# mpl.rcParams['ytick.major.width'] = 1
# mpl.rcParams['ytick.minor.size'] = 5
# mpl.rcParams['ytick.minor.width'] = 1




def sort_low(error):
 val_min=1.e8
 error_f=[]
 for i in range(len(error)-1):
   if error[i]>=error[i+1] and  error[i]<=val_min :
    error_f.append(error[i])
    val_min=error[i+1]*1.0
    pass

 return error_f


R=np.loadtxt("Es.txt")
x_memorys=R[:,2]
x_flopss=R[:,3]
y=R[:,1]
y=list(y)
errors= [ abs  ( (i+1458.164)/1458.164)   for i in y]



R=np.loadtxt("PEPS.txt")
x_memoryp=R[:,2]
y=R[:,1]
y=list(y)
errorp= [ abs  ( (i+1458.164)/1458.164)   for i in y]



R=np.loadtxt("Em.txt")
x_memorym=R[:,2]
x_flopsm=R[:,3]
y=R[:,1]
y=list(y)
errorm= [ abs  ( (i+1458.164)/1458.164)   for i in y]


R=np.loadtxt("Ef.txt")
x_memoryf=R[:,2]
x_flopsf=R[:,3]
yf=R[:,1]
yf=list(yf)
errorf= [ abs  ( (i+1458.164)/1458.164)   for i in yf]


#fig=plt.figure(figsize=(7,7))

plt.plot( x_memoryp,errorp,'+-',markersize=10, color = '#f12626', label='peps')
plt.plot( x_memorys,errors,'h-',markersize=10, color = '#400e9c', label='sMERA')
plt.plot( x_memorym,errorm,'o-',markersize=10, color = '#f57900', label='mMERA')
plt.plot(x_memoryf,errorf,'s-',markersize=10, color = '#cf729d', label='fMERA')


# plt.plot( x_flopss,errors,'h-',markersize=10, color = '#400e9c', label='sMERA')
# plt.plot( x_flopsm,errorm,'o-',markersize=10, color = '#f57900', label='mMERA')
# plt.plot(x_flopsf,errorf,'s-',markersize=10, color = '#cf729d', label='fMERA')


#plt.plot(  yMX12,'-.',lw=3,markersize=10, color = '#cf729d', label='MERA, $\chi=12$')
#plt.plot(  yMX18,'-.',lw=3,markersize=10, color = '#0e9c38', label='MERA, $\chi=18$')
#plt.plot(  yMX22,'-.',lw=3,markersize=10, color = '#400e9c', label='MERA, $\chi=22$')
#plt.plot(  yMX28,'-.',lw=3,markersize=10, color = '#f12626', label='MERA, $\chi=28$')


#plt.xscale('log')
plt.yscale('log')


#plt.title('qmps')
plt.xlabel('peak-size',fontsize=11)
#plt.xlabel('flops',fontsize=14)
plt.ylabel(r'$\delta E$',fontsize=11)
#plt.axhline(0.028,color='black', label='PEPS, D=2')
#plt.axhline(0.010,color='black', label='PEPS, D=4')
#plt.axhline(0.004,color='black', label='PEPS, D=5')




#plt.xlim([20,1000])
plt.ylim([5.e-3, 2.e-1])

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc="upper right", prop={'size': 10})




plt.grid(True)
plt.savefig('errorM.pdf')
plt.clf()
