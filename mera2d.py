import quf
import cotengra as ctg
from quimb import *
import time
from cotengra.parallel import RayExecutor
from concurrent.futures import ProcessPoolExecutor
from loky import get_reusable_executor




#client_gpu = quf.get_client_g(1.)
#client_cpu = quf.get_client_c(1.0)              # 0.5 == two tasks are perfomed on one cpu
#ray start --head --num-cpus=12

def    mera_tn_2d():

  data_type= 'float64' #"complex128"  #  'float32'
  dist_type="normal"         #{'normal', 'uniform', 'exp'}
  method="mgs"           #svd, qr, mgs, exp
  jit_fn=False
  phys_dim=2
  chi=8
  device='cpu'
  autodiff_backend="torch"
  executor =  get_reusable_executor()          #get_reusable_executor()  #   RayExecutor() #client_cpu  # #RayExecutor()  #get_reusable_executor()   #RayExecutor() #RayExecutor()  #None #ProcessPoolExecutor()
  division_seg=16               # segment number of interaction list ~ #cpus  default_remote_opts={'num_cpus': 12.0}
#ray stop --force
####################################

  opt = ctg.ReusableHyperOptimizer(
     progbar=False,
     minimize='combo-64',       #{'size', 'flops', 'combo'}, what to target
     #reconf_opts={}, 
     max_repeats=2**6,
     max_time=3600,
#    max_time='rate:1e6',
     parallel=True,
     #optlib='baytune',         # 'nevergrad', 'baytune', 'chocolate','random'
     directory="cash/",
     #executor=executor
 )
  #opt="auto-hq"


  tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale=quf.Tn_mera_build(
     phys_dim=phys_dim,
     chi=chi,
     data_type=data_type,
     dist_type=dist_type)



  save_to_disk(method,"Store/method")
############################################################



  #tn_U=load_from_disk("Store/tn_U")
  #tn_U.astype_(data_type)
  quf.change_datatype(tn_U, data_type)
  #width_max, flops_max=quf.Info_contract(tn_U,list_sites,data_type=data_type,opt=opt)  
  quf.Plot_TN(tn_U,list_scale,list_tags_I, list_tags_U,phys_dim)


  #start_time = time.time()
###############################################################
  #print ("M", quf.Mag_calc(tn_U, opt,data_type=data_type) )
  #print ( "E_init=",quf.energy_f(tn_U, list_sites, list_inter,optimize=opt),"--- %s seconds ---" % (time.time() - start_time) )
  #start_time = time.time()
  #print ( "E_init=",quf.energy_mps(tn_U, list_sites, list_inter,optimize=opt),"--- %s seconds ---" % (time.time() - start_time)  )
  print ( "chi", tn_U.max_bond() )
  #quf.expand_bond_MERA(tn_U, list_tags_I,list_tags_U, method='pad',new_bond_dim=64, rand_strength=0.00300,rand_strength_u=0.0030, data_type=data_type)    
  #tn_U=quf.TN_to_iso(tn_U, list_tags_I,list_tags_U)
  print ( "chi_new", tn_U.max_bond()  )
  #start_time = time.time()
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

  #tnopt_mera=quf.auto_diff_mera(tn_U, list_sites,list_inter , opt, optimizer_c=optimizer_c, tags=[], jit_fn=jit_fn,  device=device,autodiff_backend=autodiff_backend)
  tnopt_mera=quf.auto_diff_mera_parallel(tn_U, list_sites,list_inter , opt, optimizer_c=optimizer_c, tags=[], jit_fn=jit_fn,  device=device, executor=executor, segment=segment, autodiff_backend=autodiff_backend)


  #tnopt_mera=quf.auto_diff_umps(tn_U, list_sites,list_inter , opt, optimizer_c=optimizer_c, tags=[], jit_fn=jit_fn,  device=device)




  tn_U = tnopt_mera.optimize(n=120 ,hessp=False, ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False)
  #tn_U =  tnopt_mera.optimize_nlopt(50 ,ftol_rel= 2.220e-14)



  #tn_U = tnopt_mera.optimize_basinhopping(n=100, nhop=10, temperature=0.5 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 1, disp=False)
  #tnopt_mera.optimizer = 'TNC' 
  #tn_U = tnopt_mera.optimize( n=1000, stepmx=200, eta=0.25, maxCGit=200, accuracy=1e-12, maxfun=int(10e+8), gtol= 1e-10, disp=False)








  #tn_U.unitize_(method=method, allow_no_left_inds=True)
  #print ( "E_f=", quf.energy_f(tn_U, list_sites, list_inter,optimize=opt) )
  #print ("M", quf.Mag_calc(tn_U, opt,data_type=data_type) )
  save_to_disk( tn_U, "Store/tn_U")



  y=tnopt_mera.losses[:]
  #y_list=[  abs((y[i]-E_exact)/E_exact)  for i in range(len(y)) ]
  x_list=[ i for i in range(len(y)) ]


  file = open("Data/mera.txt", "w")
  for index in range(len(y)):
     file.write(str(x_list[index]) + "  "+ str(y[index])+ "  " + "\n")








