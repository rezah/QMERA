import quf
import cotengra as ctg
from quimb import *


def    cont_tn_2d():

  data_type='float32'
  dist_type="normal"         #{'normal', 'uniform', 'exp'}
  method="mgs"           #svd, qr, mgs, exp
  jit_fn=True
  chi=[2,4]
  device='cpu'
  phys_dim=2
  list_width_max=[]
  list_flops_max=[]
  list_peak_max=[]
  
####################################

  opt = ctg.ReusableHyperOptimizer(
     progbar=True,
     minimize='flops',       #{'size', 'flops', 'combo'}, what to target
     reconf_opts={}, 
     max_repeats=2**7,
     max_time=3600,
#    max_time='rate:1e6',
     parallel=True,
     #optlib='baytune',         # 'nevergrad', 'baytune', 'chocolate','random'
     directory="cash/"
 )
  #opt="auto-hq"




  #tn_U.unitize_(method=method, allow_no_left_inds=True)
  #tn_U=load_from_disk("Store/tn_U")
  for i in chi:
   #tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale=quf.Tn_mera_build(chi=i,data_type=data_type,dist_type=dist_type)
   tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale=quf.Tn_mera_build_3d(phys_dim=phys_dim,chi=i,data_type=data_type,dist_type=dist_type)

   #save_to_disk(method,"Store/method")
 ############################################################
   #width_max, flops_max, peak_max=quf.Info_contract(tn_U,list_sites,data_type=data_type,opt=opt)
   #list_width_max.append(width_max)
   #list_flops_max.append(flops_max)
   #list_peak_max.append(peak_max)
 ############################################################
   width_max, flops_max, peak_max=quf.Info_contract(tn_U,list_sites,data_type=data_type,opt=opt)
   list_width_max.append(width_max)
   list_flops_max.append(flops_max)
   list_peak_max.append(peak_max)
  #quf.Plot_TN(tn_U,list_scale)



###############################################################
#  print ( "E_init=",quf.energy_f(tn_U, list_sites, list_inter,optimize=opt) )
#  print ( "chi", tn_U.max_bond() )
#  #quf.expand_bond_MERA(tn_U, list_tags_I, method='pad',new_bond_dim=16, rand_strength=0.0100,rand_strength_u=0.010, data_type=data_type)    
#  #tn_U=quf.TN_to_iso(tn_U, list_tags_I,list_tags_U)
#  print ( "chi_new", tn_U.max_bond()  )
#  print ("E_init_f=",quf.energy_f(tn_U, list_sites, list_inter,optimize=opt))
#  tn_U.unitize_(method=method, allow_no_left_inds=True)
  #print ("E_init=", quf.energy_f(tn_U, list_sites, list_inter,optimize=opt))
#################################################






#  tnopt_mera=quf.auto_diff_mera(tn_U, list_sites,list_inter , opt, optimizer_c='L-BFGS-B', tags=[], jit_fn=jit_fn,  device=device)
#  tnopt_mera.optimizer = 'L-BFGS-B' 
  #tnopt_mera.optimizer = 'CG' 
  #tnopt_mera.optimizer = 'adam' 
#  tn_U = tnopt_mera.optimize(n=1000 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False)


  #tn_U = tnopt_mera.optimize_basinhopping(n=100, nhop=10, temperature=0.5 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 1, disp=False)


  #tnopt_mera.optimizer = 'TNC' 
  #tn_U = tnopt_mera.optimize( n=1000, stepmx=200, eta=0.25, maxCGit=200, accuracy=1e-12, maxfun=int(10e+8), gtol= 1e-10, disp=False)








  #tn_U.unitize_(method=method, allow_no_left_inds=True)
  #print ( "E_f=", quf.energy_f(tn_U, list_sites, list_inter,optimize=opt) )
  #save_to_disk( tn_U, "Store/tn_U")



  #y=tnopt_mera.losses[:]
  #y_list=[  abs((y[i]-E_exact)/E_exact)  for i in range(len(y)) ]
  #x_list=[ i for i in range(len(y)) ]


#  file = open("Data/mera.txt", "w")
#  for index in range(len(y)):
#     file.write(str(x_list[index]) + "  "+ str(y[index])+ "  " + "\n")


  #print ( "chi=",chi, "width=",list_width_max, "flops=",list_flops_max, "peak_max", list_peak_max)
  #file = open("Data/contractInfo.txt", "w")
  #for index in range(len(chi)):
  #   file.write( str(chi[index]) + "  " +str(list_width_max[index]) + "  "+ str(list_flops_max[index])
  #   +"  "+str(list_peak_max[index])+ "  " + "\n")
  print ( "chi=",chi, "width=",list_width_max, "flops=",list_flops_max, "peak_max", list_peak_max)
  file = open("Data/contractInfo.txt", "w")
  for index in range(len(chi)):
     file.write( str(chi[index]) + "  " +str(list_width_max[index]) + "  "+ str(list_flops_max[index])
     +"  "+str(list_peak_max[index])+ "  " + "\n")









