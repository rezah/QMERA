
#watch -n 2 nvidia-smi
#watch -n 3 nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,pstate,utilization.gpu,utilization.memory --format=csv
#CUDA_VISIBLE_DEVICES=3
import torch
import sys
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION', )
from subprocess import call
# call(["nvcc", "--version"]) does not work
#! nvcc --version
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())



print (torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device(0),torch.cuda.device_count(),torch.cuda.get_device_name(0))




















def  update_neighbour_Uni(tn_U, cor_origin,common_inds, new_bond_dim,rand_strength,rand_strength_u,tags_u,method="pad",data_type="float64"):

     #print (common_inds)
     new_bond_dim_local=tn_U.ind_size(common_inds[0]) 
     neighbour=tn_U.ind_map[common_inds[0]]-cor_origin 
     tensor=tn_U.tensor_map[ list(neighbour)[0] ]
     right_inds = list( set(tensor.inds) - set(tensor.left_inds) )
     left_inds = list( set(tensor.left_inds) )


     if len(set(tensor.left_inds))==len(right_inds):      
              send_out_ind=0
              if left_inds[0]==common_inds[0]:
                    inds_to_expand =  common_inds+[right_inds[0]]
                    send_out_ind=[right_inds[0]]
              if left_inds[1]==common_inds[0]:
                    inds_to_expand =  common_inds+[right_inds[1]]
                    send_out_ind=[right_inds[1]]

              chi_o=np.prod([ tn_U.ind_size(i)  for i in inds_to_expand])
              chi_n=new_bond_dim_local
              
              shape_o=[ tn_U.ind_size(i)  for i in left_inds]+[chi_n]
              
              pads = [(0, 0) if i not in inds_to_expand else
                             (0, max(new_bond_dim_local - d, 0))
                             for d, i in zip(tensor.shape, tensor.inds)]

              

              if rand_strength_u > 0:
                  edata = do('pad', tensor.data, pads, mode=rand_padder,
                           rand_strength=rand_strength_u)
              else:
                 edata = do('pad', tensor.data, pads, mode='constant')


              tensor.modify(data=edata)
              tensor.modify(left_inds=left_inds)

              return send_out_ind, neighbour

     else:
             for i in tensor.tags:
                   tags_u.append(i)
             inds_to_expand = common_inds
             pads = [(0, 0) if i not in inds_to_expand else
                            (0, max(new_bond_dim_local - d, 0))
                            for d, i in zip(tensor.shape, tensor.inds)]
             if method=="pad" :
                   if rand_strength > 0:
                      edata = do('pad', tensor.data, pads, mode=rand_padder,
                                 rand_strength=rand_strength)
                   else:
                      edata = do('pad', tensor.data, pads, mode='constant')
             elif method=="manual":
                   edata = do('pad', tensor.data, pads, mode='constant', constant_values=rand_strength)


             tensor.modify(data=edata)
             tensor.modify(left_inds=left_inds)
             ##print ("result", pads,tn_U.tensor_map[neighbour], "\n")
             return [],0











def    mera_tn_3d():

  opt = ctg.ReusableHyperOptimizer(
      methods=['greedy', 'kahypar'],
      max_repeats=32,
      max_time='rate:1e6',
      parallel=True,
      reconf_opts={},
      progbar=True,
      directory="cash/")

  opt = ctg.ReusableHyperOptimizer(
     progbar=True,
     reconf_opts={},
     max_repeats=32,
     parallel=True,
     directory="cash/")


  opt="auto-hq"

  data_type='float64'
  Num_layers = 2
  N_x=3*2
  N_y=3*2
  N_z=3*2
  tn_3d=TN3D_rand(N_x, N_y, N_z, 2, cyclic=False, site_tag_id='g{},{},{}', dtype='float64')
  tn_U = TN3DUni.empty(N_x, N_y, N_z, phys_dim=2)
  chi = 8


  print ("N_x, N_y, N_z", N_x, N_y,N_z, "chi", chi)
  list_sites, list_inter=Heis_local_Ham_open_3D(N_x,N_y,N_z)
  list_sites, list_inter=Heis_local_Ham_cycle_3d(N_x,N_y,N_z)
  list_tags_U=[]
  list_tags_I=[]
  shared_U=[]
  shared_I=[]
  list_scale=[]
  num_layer=[0,0,0,0]     #U, I, ScaleU, ScaleI


#  Iso_44_11_3D(tn_U, N_x, N_y, N_z,chi,num_layer,list_tags_U,list_tags_I,uni="on",last_bond="on", cycle="on", Iso="on")
  #Iso_44_11_3D(tn_U, N_x//4, N_y//4, N_z//4,chi,num_layer,list_tags_U,list_tags_I,uni="on",last_bond="on", cycle="off")
  #Iso_22_3D(tn_U, N_x//4, N_y//4,N_z//4,num_layer,list_tags_I,chi,seed_val=10,last_bond="on")
#  num_layer[3]+=1

  Iso_33_11_3D(tn_U, N_x, N_y, N_z,chi,num_layer,list_tags_U,list_tags_I,uni="on",last_bond="off", cycle="off", Iso="on")
  Iso_22_3D(tn_U, N_x//3, N_y//3,N_z//3,num_layer,list_tags_I,chi,seed_val=10,last_bond="on")
  num_layer[3]+=1
  


  #tn_U.astype_(data_type)
  check_tags(tn_U, list_tags_I, list_tags_U)
  #list_scale=eliminate_dupl(list_scale)


  print (num_layer)
  #Info_contract(tn_U,list_sites,data_type=data_type,opt=opt)
  Plot_TN_3d(tn_U,num_layer,tn_3d)




  #tn_U=load_from_disk("Store/tn_U")
  ##print ("E_init=",energy_f(tn_U, list_sites, list_inter,optimize=opt))
  expand_bond_MERA(tn_U, list_tags_I, new_bond_dim=16, rand_strength=0.01)    
  #tn_U.unitize_(method='mgs', allow_no_left_inds=True)
  print ( tn_U.max_bond()  )
  ##print ("E_init=",energy_f(tn_U, list_sites, list_inter,optimize=opt))

  #tn_U.unitize_(method='mgs', allow_no_left_inds=True)
  ##print ("E_init=",energy_f(tn_U, list_sites, list_inter,optimize=opt))


  tnopt_mera=auto_diff_mera(tn_U, list_sites,list_inter , opt, optimizer_c='L-BFGS-B')
  tnopt_mera.optimizer = 'L-BFGS-B' 
  tn_U = tnopt_mera.optimize(n=5000 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False)
  save_to_disk(tn_U, "Store/tn_U")


  y=tnopt_mera.losses[:]
  #y_list=[  abs((y[i]-E_exact)/E_exact)  for i in range(len(y)) ]
  x_list=[ i for i in range(len(y)) ]

  file = open("Data/mera.txt", "w")
  for index in range(len(y)):
     file.write(str(x_list[index]) + "  "+ str(y[index])+ "  " + "\n")











#tn_U.astype_(data_type)

#def    mera_tn_2d_tree():

#  opt = ctg.ReusableHyperOptimizer(
#      methods=['greedy', 'kahypar'],
#      max_repeats=32,
#      max_time='rate:1e6',
#      parallel=True,
#      reconf_opts={},
#      progbar=True,
#      directory="cash/"
#  )


#  opt = ctg.ReusableHyperOptimizer(
#     progbar=True,
#     reconf_opts={},
#     max_repeats=32,
#     parallel=True,
#     directory="cash/"
# )


#  #opt="auto-hq"

#  data_type='float64'
#  num_layers = 2
#  N_x=3**num_layers
#  N_y=3**num_layers
#  N_x=4*2
#  N_y=4*2
#  tn_U = quf.TN2DUni.empty(N_x, N_y, phys_dim=2,data_type=data_type)
#  chi = 8



#  #print ("N_x, N_y", N_x, N_y, "chi", chi)
#  list_sites, list_inter = quf.Heis_local_Ham_open(N_x,N_y,data_type=data_type)
#  list_tags_U=[]
#  list_tags_I=[]
#  shared_U=[]
#  shared_I=[]
#  list_scale=[]
#  num_layer=[0,0]     #U, I
#  label_list=[0]





#  quf.Iso_44_11(tn_U, N_x, N_y,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,scale=0,uni="on",last_bond="off",cycle="off",
#        data_type=data_type,Iso_apply="on", Iso_1=True, Iso_2=False)

#  quf.Iso_22(tn_U, N_x//4, N_y//4,chi,num_layer, list_tags_I,label_list,shared_I,list_scale,scale=1,seed_val=10, last_bond="on",data_type=data_type, Iso=False)



  #quf.Iso_33_11(tn_U, N_x, N_y,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,scale=0,uni="on",last_bond="off",cycle="off",data_type=data_type, Iso=False)

  #quf.Iso_33_11(tn_U, N_x//3, N_y//3,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,scale=1,uni="off",last_bond="on",cycle="off",data_type=data_type, Iso=False)


  #tn_U=quf.Init_TN(tn_U, list_tags_I, list_tags_U)



  #tn_U.astype_(data_type)
  quf.check_tags(tn_U, list_tags_I, list_tags_U)
  list_scale=quf.eliminate_dupl(list_scale)


  #tn_U=quf.norm_f_tree(tn_U,optimize=opt)

  #quf.Info_contract_Tree(tn_U,list_sites,data_type=data_type,opt=opt)
  quf.Plot_TN(tn_U,list_scale)
  


  
  #print ("E_init=",energy_f(tn_U, list_sites, list_inter,optimize=opt))
  #quf.expand_bond_MERA_tree(tn_U, list_tags_I, new_bond_dim=12, rand_strength=0.01)    
  #tn_U.unitize_(method='mgs', allow_no_left_inds=True)
  #print ( tn_U.max_bond()  )
  #print ("E_init=",quf.energy_f_tree(tn_U, list_sites, list_inter,optimize=opt))
  #print ("E_init=",quf.energy_f(tn_U, list_sites, list_inter,optimize=opt))

  #tn_U.unitize_(method='mgs', allow_no_left_inds=True)
  #print ("E_init=",energy_f(tn_U, list_sites, list_inter,optimize=opt))

  tnopt_mera=quf.auto_diff_mera_tree(tn_tree, list_sites,list_inter , opt, optimizer_c='L-BFGS-B')
  tnopt_mera.optimizer = 'L-BFGS-B' 
  #tnopt_mera.optimizer = 'CG' 
  tn_tree = tnopt_mera.optimize(n=10 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False)
  save_to_disk(tn_U, "Store/tn_U")


  y=tnopt_mera.losses[:]
  #y_list=[  abs((y[i]-E_exact)/E_exact)  for i in range(len(y)) ]
  x_list=[ i for i in range(len(y)) ]

  file = open("Data/mera.txt", "w")
  for index in range(len(y)):
     file.write(str(x_list[index]) + "  "+ str(y[index])+ "  " + "\n")






################################################3D lattice-9*9*9#######################
#  Num_layers = 2
#  N_x=3*3
#  N_y=3*3
#  N_z=3*3
#  tn_3d=TN3D_rand(N_x, N_y, N_z, 2, cyclic=False, site_tag_id='g{},{},{}', dtype='float64')
#  tn_U = TN3DUni.empty(N_x, N_y, N_z, phys_dim=2)


#  chi=64
#  num_layer=0
#  list_tags_U=[]
#  list_tags_I=[]
#  num_layer=[0,0]  #U, I
#  list_sites, list_inter=Heis_local_Ham_open_3D(N_x,N_y,N_z)



#  num_layer,list_tags_U,list_tags_I=Iso_33_11_3D(tn_U, N_x, N_y, N_z,chi,num_layer,list_tags_U,list_tags_I,uni="on",last_bond="off",cycle="on")
#  num_layer,list_tags_U,list_tags_I=Iso_33_11_3D(tn_U, N_x//3, N_y//3, N_z//3,chi,num_layer,list_tags_U,list_tags_I,uni="on",last_bond="on",cycle="on")
#############################################################################################################3

















#Num_layers = 3
#N_x=4**Num_layers
#N_y=4**Num_layers
#tn_U = TN2DUni.empty(N_x, N_y, phys_dim=2)
#chi = 8

#print ("N_x, N_y", N_x, N_y, "chi", chi)
#list_sites, list_inter = Heis_local_Ham_open(N_x,N_y)

#for i in range(Num_layers ):
# N_x_l=N_x//(4**i)
# N_y_l=N_y//(4**i)
# if i<Num_layers-1:
#  Iso_44_11(tn_U, N_x//(4**i), N_y//(4**i), chi, uni="on", last_bond="off")
# else:
#  Iso_44_11(tn_U, N_x//(4**i), N_y//(4**i),chi, uni="off", last_bond="on")




#####################################44--->33---->22--->11#################################
#num_layers = 2
#N_x=3**num_layers
#N_y=3**num_layers
#tn_U = quf.TN2DUni.empty(N_x, N_y, phys_dim=2)
#chi = 12


#print ("N_x, N_y", N_x, N_y, "chi", chi)
#list_sites, list_inter = quf.Heis_local_Ham_open(N_x,N_y)
#list_tags_U=[]
#list_tags_I=[]
#num_layer=[0,0]  #U, I



#for i in range(num_layers):
# N_x_l=N_x//(3**i)
# N_y_l=N_x//(3**i)
# if i<num_layers-1:
#  quf.Iso_33_11(tn_U, N_x_l, N_y_l,chi,num_layer,list_tags_U,list_tags_I,uni="on",last_bond="off",cycle="on")
# elif i==num_layers-1:
#  quf.Iso_33_11(tn_U, N_x_l, N_y_l,chi,num_layer,list_tags_U,list_tags_I,uni="on",last_bond="on",cycle="on")

#    
#tn_U.unitize_(method='mgs', allow_no_left_inds=True)
#quf.check_tags(tn_U, list_tags_I, list_tags_U)
##print ( (tn_U.H & tn_U)^all )




#  num_layers = 2
#  N_x=4**num_layers
#  N_y=4**num_layers
#  tn_U = TN2DUni.empty(N_x, N_y, phys_dim=2)
#  chi = 8


#  print ("N_x, N_y", N_x, N_y, "chi", chi)
#  list_sites, list_inter = Heis_local_Ham_open(N_x,N_y)

#  for i in range(num_layers):
#   N_x_l=N_x//(4**i)
#   N_y_l=N_x//(4**i)

#   if  i<=num_layers-2:
#   #uni_xy_33(tn_U, N_x_l, N_y_l, i, chi,seed_val=0, cycle="off")
#    uni_xy_44(tn_U, N_x_l, N_y_l, i, chi,seed_val=0, cycle="off")

#      #uni_diag_33(tn_U, N_x_l, N_y_l, i, chi,seed_val=0, cycle="off")

#   #Iso_22(tn_U, N_x_l, N_y_l, i, chi, num_layers,seed_val=10)
#   Iso_44(tn_U, N_x_l, N_y_l, i, chi, num_layers,seed_val=10)
 
 
#####################################1d Ternary MERA#################################

#  Num_layers = 2
#  N_x=18
#  N_y=18
#  tn_U = TN2DUni.empty(N_x, N_y, phys_dim=2)
#  chi = 2

#  print ("N_x, N_y", N_x, N_y, "chi", chi)
#  list_sites, list_inter = Heis_local_Ham_open(N_x,N_y)

#  uni_dense_22(tn_U, N_x, N_y, 0,chi, seed_val=10, cycle="off")


#  uni_local_y_2(tn_U, N_x, N_y, 1,chi, seed_val=10, cycle="on", shift_x=0,shift_y=2, dis_opt=3)
#  Iso_33_y(tn_U, N_x, N_y,0,chi,seed_val=10,last_bond="off")


#  uni_local_x_2(tn_U, N_x, N_y//3, 1,chi, seed_val=10, cycle="on", shift_x=0,shift_y=2, dis_opt=3)
#  Iso_33_x(tn_U, N_x, N_y//3,0,chi,seed_val=10,last_bond="off")

#  N_x=6
#  N_y=6


#  uni_local_y_2(tn_U, N_x, N_y, 1,chi, seed_val=10, cycle="on", shift_x=0,shift_y=2, dis_opt=3)
#  Iso_33_y(tn_U, N_x, N_y,0,chi,seed_val=10,last_bond="off")


#  uni_local_x_2(tn_U, N_x, N_y//3, 1,chi, seed_val=10, cycle="on", shift_x=0,shift_y=2, dis_opt=3)
#  Iso_33_x(tn_U, N_x, N_y//3,0,chi,seed_val=10,last_bond="off")


#  Iso_22(tn_U, N_x//3, N_y//3,2,chi,seed_val=10,last_bond="on")


##################################### modified binary MERA#################################


#  Num_layers = 2
#  N_x=16
#  N_y=16
#  tn_U = TN2DUni.empty(N_x, N_y, phys_dim=2)
#  chi = 8

#  print ("N_x, N_y", N_x, N_y, "chi", chi)
#  list_sites, list_inter = Heis_local_Ham_open(4,4)

#  uni_local_y_2(tn_U, N_x, N_y, 0,chi, seed_val=10, cycle="on", shift_x=0,shift_y=3, dis_opt=4)
#  Iso_22_y(tn_U, N_x, N_y,0,chi,seed_val=10,last_bond="off")
#  uni_local_y_2(tn_U, N_x, N_y//2, 1,chi, seed_val=10, cycle="on", shift_x=0,shift_y=1, dis_opt=4)
#  Iso_22_y(tn_U, N_x, N_y//2,1,chi,seed_val=10,last_bond="off")

#  N_x=16
#  N_y=4


#  uni_local_x_2(tn_U, N_x, N_y, 0,chi, seed_val=10, cycle="on", shift_x=3,shift_y=0, dis_opt=4)
#  Iso_22_x(tn_U, N_x, N_y,0,chi,seed_val=10,last_bond="off")
#  uni_local_x_2(tn_U, N_x//2, N_y, 1,chi, seed_val=10, cycle="on", shift_x=1,shift_y=0, dis_opt=4)
#  Iso_22_x(tn_U, N_x//2, N_y,1,chi,seed_val=10,last_bond="off")


#  N_x=4
#  N_y=4

#  Iso_44(tn_U, N_x, N_y,2,chi,seed_val=10,last_bond="on")

#  Num_layers = 2
#  N_x=3**Num_layers
#  N_y=3**Num_layers
#  tn_U = TN2DUni.empty(N_x, N_y, phys_dim=2)
#  chi = 8

#  print ("N_x, N_y", N_x, N_y, "chi", chi)
#  list_sites, list_inter = Heis_local_Ham_open(N_x,N_y)

#  Iso_33_11(tn_U, N_x, N_y,chi,uni="on",last_bond="off")
#  Iso_33_11(tn_U, N_x//3, N_y//3,chi,uni="off",last_bond="on")
