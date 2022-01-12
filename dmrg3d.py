import quf
import cotengra as ctg
from quimb import *
import quimb.tensor as qtn
from math import exp, pi, sin, cos, acos, log#, polar

def    dmrg():

  D=200
  iter_n=10
  data_type='float64'
  L_x=8
  L_y=8
  #L_z=8
  #L_L=L_x*L_y*L_z
  L_L=L_x*L_y
  #MPO_origin=quf.mpo_3d_His(L_x,L_y,L_z,L_L, data_type=data_type,chi=300,cutoff_val=1.0e-12)
  #MPO_origin=quf.mpo_2d_His_long(L_x,L_y,L_L, data_type=data_type,cutoff_val=1.0e-14,chi=300, alpha=3, phi=pi/6,theta=pi/6, N_interval=1000)
  #save_to_disk(MPO_origin,"Store/MPO_origin")
  MPO_origin=load_from_disk("Store/MPO_origin")
  p_init=load_from_disk("Store/p_dmrg")
  print (  MPO_origin.show()  )

  #L_L=32
  #MPO_origin=qtn.MPO_ham_heis(L=L_L, j=(1.0,1.0,1.0), bz=0.0, S=0.5, cyclic=False)


  dmrg = qtn.DMRG2(
  MPO_origin, 
  bond_dims=[D], 
  cutoffs=1.e-12,
  #p0=p_init
  ) 
  dmrg.solve( tol=1.e-10, verbosity=2, max_sweeps=iter_n )
  E_exact=dmrg.energy
  p_dmrg=dmrg.state
  save_to_disk(p_dmrg,"Store/p_dmrg")
  print( "DMRG", E_exact, p_dmrg.show(), E_exact/L_L)
  print (dmrg.energies)
  y=dmrg.energies

  #tn_U=load_from_disk("Store/tn_U")
  file = open("Data/dmrg.txt", "w")
  for index in range(len(y)):
     file.write(str(index) + "  "+ str(y[index])+ "  " + "\n")



