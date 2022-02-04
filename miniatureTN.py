import quf
import cotengra as ctg
from quimb import *
import quimb as qu
import time
from cotengra.parallel import RayExecutor
from concurrent.futures import ProcessPoolExecutor
from loky import get_reusable_executor


def    miniature_tn_():

  data_type='float64'
  dist_type="normal"         #{'normal', 'uniform', 'exp'}
  method="mgs"           #svd, qr, mgs, exp
  jit_fn=False
  phys_dim=2
  chi=5
  device='cpu'
  autodiff_backend="torch"
  executor =  RayExecutor()            #get_reusable_executor()  #   RayExecutor() #client_cpu  # #RayExecutor()  #get_reusable_executor()   #RayExecutor() #RayExecutor()  #None #ProcessPoolExecutor()
  division_seg=16               # segment number of interaction list ~ #cpus
#ray stop --force

####################################

  opt = ctg.ReusableHyperOptimizer(
     progbar=True,
     minimize="combo-64", #  "combo-64",       #{'size', 'flops', 'combo'}, what to target
     #reconf_opts={}, 
     max_repeats=2**7,
     max_time=3600,
#    max_time='rate:1e6',
     parallel=True,
     #optlib='baytune',         # 'nevergrad', 'baytune', 'chocolate','random'
     directory="cash/"
 )
  #opt="auto-hq"


  #tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale=quf.dMERA_build(phys_dim=phys_dim,chi=chi,data_type=data_type,dist_type=dist_type)
  #tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale=quf.MiniatureTN_build(phys_dim=phys_dim,chi=5,chi_p=10,cycle_u="False",data_type=data_type,dist_type=dist_type)
  #tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale=quf.MiniatureTN_build_four(phys_dim=phys_dim,chi=5,chi_p=6,chi_pp=6,cycle_u="False",data_type=data_type,dist_type=dist_type)
  #tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale=quf.MiniatureTN_build_three(phys_dim=phys_dim,chi=3,chi_p=8,chi_pp=7,depth_U=2,cycle_u="False",data_type=data_type,dist_type=dist_type)

  tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale=quf.MiniatureTN_build_four_2d(phys_dim=phys_dim,chi=12,chi_p=5,chi_pp=5,
                      cycle_u="False",data_type=data_type,dist_type=dist_type)
 
 

  save_to_disk(method,"Store/method")
############################################################



  
  #tn_U=load_from_disk("Store/tn_U")
  #tn_U.astype_(data_type)
  #quf.change_datatype(tn_U, data_type)
  #width_max, flops_max=quf.Info_contract(tn_U,list_sites,data_type=data_type,opt=opt)  
  quf.Plot_TN_3d_Miniature(tn_U,list_scale,list_tags_I, list_tags_U,phys_dim)


  start_time = time.time()
###############################################################
  #print ("M", quf.Mag_calc(tn_U, opt,data_type=data_type) )
  print ( "E_init=",quf.energy_f(tn_U, list_sites, list_inter,optimize=opt),"--- %s seconds ---" % (time.time() - start_time) )
  #start_time = time.time()
  #print ( "E_init=",quf.energy_mps(tn_U, list_sites, list_inter,optimize=opt),"--- %s seconds ---" % (time.time() - start_time)  )
  #print ( "chi", tn_U.max_bond() )

  #quf.expand_bond_MERA(tn_U, list_tags_I,list_tags_U, method='pad',new_bond_dim=12, rand_strength=0.00300,rand_strength_u=0.0030, data_type=data_type)    
  #quf.expand_bond_Miniat(tn_U, list_tags_I,list_tags_U, method='pad',new_bond_dim=6, new_bond_dim_internal=7, rand_strength=0.00300,rand_strength_u=0.0030, data_type=data_type)    
  #quf.expand_bond_TN(tn_U, list_tags_I,list_tags_U, method='pad',new_bond_dim=5,chi_check=[12], rand_strength=0.00300,rand_strength_u=0.0030, data_type=data_type)    
  #quf.expand_bond_TN(tn_U, list_tags_I,list_tags_U, method='pad',new_bond_dim=4,chi_check=[], rand_strength=0.00300,rand_strength_u=0.0030, data_type=data_type)    
  #quf.expand_bond_TN(tn_U, list_tags_I,list_tags_U, method='pad',new_bond_dim=5,chi_check=[], rand_strength=0.00300,rand_strength_u=0.0030, data_type=data_type)    
  #quf.Plot_TN_3d_Miniature(tn_U,list_scale,list_tags_I, list_tags_U,phys_dim)




  #tn_U=quf.TN_to_iso(tn_U, list_tags_I,list_tags_U)
  print ( "chi_new", tn_U.max_bond()  )
  start_time = time.time()
  #print ("E_init_f=",quf.energy_f(tn_U, list_sites, list_inter,optimize=opt), "--- %s seconds ---" % (time.time() - start_time))
  #print ( "E_init=",quf.energy_mps(tn_U, list_sites, list_inter,optimize=opt) )





#  tn_U.unitize_(method=method, allow_no_left_inds=True)
  #print ("E_init=", quf.energy_f(tn_U, list_sites, list_inter,optimize=opt))
#################################################


  segment=len(list_inter)//division_seg #//12

  optimizer_c='L-BFGS-B'
  optimizer_c='adam'
  #optimizer_c='CG'

  #optimizer_c='LD_VAR2'
  #optimizer_c='LD_LBFGS'
  #optimizer_c='LD_TNEWTON_PRECOND_RESTART'
  #optimizer_c='LN_COBYLA'

  tnopt_mera=quf.auto_diff_mera(tn_U, list_sites,list_inter , opt, optimizer_c=optimizer_c, tags=[], jit_fn=jit_fn,  device=device)
  #tnopt_mera=quf.auto_diff_umps(tn_U, list_sites,list_inter , opt, optimizer_c=optimizer_c, tags=[], jit_fn=jit_fn,  device=device)
  #tnopt_mera=quf.auto_diff_mera_parallel(tn_U, list_sites,list_inter , opt, optimizer_c=optimizer_c, tags=[], jit_fn=jit_fn,  device=device, executor=executor, segment=segment, autodiff_backend=autodiff_backend)




  tn_U = tnopt_mera.optimize(n=20 ,hessp=False, ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False)
  #tn_U =  tnopt_mera.optimize_nlopt(400 ,ftol_rel= 2.220e-14)

  #print ( "Tensor", tn_U[list_tags_I[0]], tn_U[list_tags_I[0]].copy() )


  #tn_U = tnopt_mera.optimize_basinhopping(n=100, nhop=10, temperature=0.5 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 1, disp=False)
  #tnopt_mera.optimizer = 'TNC' 
  #tn_U = tnopt_mera.optimize( n=1000, stepmx=200, eta=0.25, maxCGit=200, accuracy=1e-12, maxfun=int(10e+8), gtol= 1e-10, disp=False)








  #tn_U.unitize_(method=method, allow_no_left_inds=True)
  #print ( "E_f=", quf.energy_f(tn_U, list_sites, list_inter,optimize=opt) )
  #print ("M", quf.Mag_calc(tn_U, opt,data_type=data_type) )
  save_to_disk( tn_U, "Store/tn_U")



  E_exact = qu.heisenberg_energy(tn_U.Lx)
  print ("E_exact", E_exact)
  y=tnopt_mera.losses[:]
  error_list=[  abs((y[i]-E_exact)/E_exact)  for i in range(len(y)) ]
  print ("E_exact=",E_exact,"error=", error_list[-1])
  x_list=[ i for i in range(len(y)) ]


  file = open("Data/mera.txt", "w")
  for index in range(len(y)):
     file.write(str(x_list[index]) + "  "+ str(y[index])+ "  "+str(error_list[index])+ "  " + "\n")










