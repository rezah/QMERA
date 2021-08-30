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



R=np.loadtxt("error.txt")
x=R[:,0]
y=R[:,1]
y= [   abs ( ((i/(8.*8.))+0.619040)/0.619040 )            for i in y]
x= [   abs ( 1./i )            for i in x]



R=np.loadtxt("errorTree.txt")
xt=R[:,0]
yt=R[:,1]
yt= [   abs ( ((i/(8.*8.))+0.619040)/0.619040 )            for i in yt]
xt= [   abs ( 1./i )            for i in xt]




fig=plt.figure(figsize=(7,7))

plt.plot( x, y, 'o',markersize=10, color = '#a40000', label='MERA')
plt.plot( xt, yt, 's',markersize=10, color = '#f57900', label='Tree')


plt.yscale('log')


#plt.title('qmps')
plt.xlabel(r'$\chi^{-1}$',fontsize=18)
plt.ylabel(r'$\delta$ E',fontsize=21)
plt.axhline(   (((-39.312834)/(8.*8.))+0.619040)/0.619040 ,color='#204a87', label='PEPS, D=4')
plt.axhline(   0.00003 ,color='#204a87', label='PEPS, D=8')


plt.xlim([0.001, 0.08])
plt.ylim([ 0.00001, 0.1])

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="center right", prop={'size': 18})




plt.grid(True)
plt.savefig('error.pdf')
plt.clf()
