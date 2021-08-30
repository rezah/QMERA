import quf
import cotengra as ctg
from quimb import *


def    tree_tn_2d():

  data_type='float64'
  dist_type="exp"         #{'normal', 'uniform', 'exp'}
  method="mgs"
  jit_fn=True
  device='cpu'
  chi=4


  opt = ctg.ReusableHyperOptimizer(
     progbar=True,
     minimize='flops',       #{'size', 'flops', 'combo'}, what to target
     reconf_opts={}, 
     max_repeats=64,
     max_time=3600,
#      max_time='rate:1e6',
     parallel=True,
     #optlib='baytune',         # 'nevergrad', 'baytune', 'chocolate','random'
     directory="cash/"
 )
  #opt="auto-hq"



####################################
  tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale=quf.Tn_tree_build(chi=chi,data_type=data_type,dist_type=dist_type)
  save_to_disk(method, "Store/method")
  #print (list_tags_U)
######################################################################################################################3







  #tn_U=load_from_disk("Store/tn_U_tree")
  #quf.Info_contract(tn_U,list_sites,data_type=data_type,opt=opt)
  #quf.Plot_TN(tn_U,list_scale)




  print ( "E_init=",quf.energy_f_tree(tn_U, list_sites, list_inter,optimize=opt) )
  print ( tn_U.max_bond() )
  #quf.expand_bond_MERA(tn_U, list_tags_I,list_tags_U, method='pad',new_bond_dim=6, rand_strength=0.0100,rand_strength_u=0.010, data_type=data_type)    
  #tn_U=quf.TN_to_iso(tn_U, list_tags_I,list_tags_U)
  #print ( tn_U.max_bond()  )
  #print ("E_init=",quf.energy_f_tree(tn_U, list_sites, list_inter,optimize=opt))
  tn_U.unitize_(method=method, allow_no_left_inds=True)
  #print ("E_init=", quf.energy_f_tree(tn_U, list_sites, list_inter,optimize=opt))




  tnopt_mera=quf.auto_diff_tree(tn_U, list_sites,list_inter , opt, optimizer_c='L-BFGS-B', tags=[],jit_fn=jit_fn,  device=device)
  tnopt_mera.optimizer = 'L-BFGS-B' 
  #tnopt_mera.optimizer = 'CG' 
  tn_U = tnopt_mera.optimize(n=1200 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False)
  #tn_U.unitize_(method=method, allow_no_left_inds=True)
  print ( "E_f=",quf.energy_f_tree(tn_U, list_sites, list_inter,optimize=opt) )
  save_to_disk(tn_U, "Store/tn_U_tree")








  y=tnopt_mera.losses[:]
  #y_list=[  abs((y[i]-E_exact)/E_exact)  for i in range(len(y)) ]
  x_list=[ i for i in range(len(y)) ]

  file = open("Data/mera.txt", "w")
  for index in range(len(y)):
     file.write(str(x_list[index]) + "  "+ str(y[index])+ "  " + "\n")









