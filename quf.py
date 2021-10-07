from quimb import *
from quimb.tensor import *
from quimb.tensor.tensor_core import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import itertools
from operator import add
import operator
import matplotlib.pyplot as plt
import math
import cmath
from numpy.linalg import inv
from cmath import exp, pi, sin, cos, acos, log, polar
import cotengra as ctg
import copy
import autoray
from progress.bar import Bar
import tqdm
import warnings
from collections import Counter
from quimb.tensor.tensor_2d import *
from quimb.tensor.tensor_3d import *
import torch
import os, sys
import scipy.linalg
from autoray import  astype
be_verbose = True
method_norm=load_from_disk("Store/method")




def change_datatype(TN, dtype):
   for t in TN:
            if t.dtype != dtype:
               t.modify(apply=lambda data: astype(data, dtype), left_inds=t.left_inds)

   l_type=[  t.dtype for t in TN ] 
   print ( "TN_data_type="   ,  l_type[0],l_type[-1]   )




def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        try:
            U, S, V = torch.svd(A)
        except:
            if be_verbose:
                print('trouble in torch gesdd routine, falling back to gesvd')
            U, S, V = scipy.linalg.svd(A.detach().numpy(), full_matrices=False, lapack_driver='gesvd')
            U = torch.from_numpy(U)
            S = torch.from_numpy(S)
            V = torch.from_numpy(V.T)

        # make SVD result sign-consistent across multiple runs
        for idx in range(U.size()[1]):
            if max(torch.max(U[:,idx]), torch.min(U[:,idx]), key=abs) < 0.0:
                U[:,idx] *= -1.0
                V[:,idx] *= -1.0

        self.save_for_backward(U, S, V)
        return U, S, V.t()

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        dV = dV.t()
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        #G.diagonal().fill_(np.inf)
        #G = 1/G
        G = safe_inverse(G)
        G.diagonal().fill_(0)

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt
        if (M>NS):
            #dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU*safe_inverse(S)) @ Vt
        if (N>NS):
            #dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
            dA = dA + (U*safe_inverse(S)) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)

        return dA

autoray.register_function('torch', 'linalg.svd', SVD.apply)








class TN2DUni(TensorNetwork2DVector,
              TensorNetwork2D,
              TensorNetwork):
    _EXTRA_PROPS = (
        '_site_tag_id',
        '_Lx',
        '_Ly',
        '_row_tag_id',
        '_col_tag_id',
        '_site_ind_id',
        '_layer_ind_id',
    )
        
    @classmethod
    def empty(
        cls, 
        Lx, 
        Ly,
        phys_dim=2,
        data_type='float64',
        site_tag_id="I{},{}",
        site_ind_id="k{},{}",
        layer_ind_id="l{},{}",
    ):
        self = object.__new__(cls)
        self._Lx = Lx
        self._Ly = Ly
        self._site_tag_id = site_tag_id
        self._site_ind_id = site_ind_id
        self._layer_ind_id = layer_ind_id
        self._row_tag_id = None
        self._col_tag_id = None
        
        
        ts = [
            Tensor(data=do("eye", phys_dim,dtype=data_type),
                   inds=[self.site_ind(i, j), 
                         self.layer_ind(i, j)],
                   tags=[self.site_tag(i, j), f"reg{i},{j}", "const"],
                   left_inds=(self.site_ind(i, j),))
             for i, j in itertools.product(range(Lx), range(Ly))
        ]
        
        super().__init__(self, ts)


        return self
    @property
    def layer_ind_id(self):
        return self._layer_ind_id
    def layer_ind(self, i, j):
        return self._layer_ind_id.format(i, j)
    def reverse_gate(
        self,
        G,
        where,
        new_sites=None,
        tags=None,
        iso=True,
    ):
        """
        Parameters
        ----------
        G : array_like
        where : tuple[tuple[int]]
        new_sites : None or tuple[tuple[int]], optional
        tags : str or sequence of str, optional
        iso : bool, optional
        """
        tags = tags_to_oset(tags)
        
        if is_lone_coo(where):
            where = (where,)
        else:
            where = tuple(where)
            
        nbelow = len(where)
        layer_ix = [self.layer_ind(i, j) for i, j in where]
        bnds = [rand_uuid() for _ in range(nbelow)]
        reindex_map = dict(zip(layer_ix, bnds))
        ##print ("reindex_map", reindex_map)
        nabove = len(G.shape) - nbelow
        if new_sites is None:
            new_sites = where[:nabove]
        new_layer_ix = [self.layer_ind(i, j) for i, j in new_sites]
        
        ##print ("new_sites", new_sites)
        ##print ("new_layer_ix", new_layer_ix)
        ##print ("layer_ix", new_layer_ix)

        #get tag of any tensor that has index "layer_ix"=[self.layer_ind(i, j) for i, j in where]
        old_tags = oset_union(t.tags for t in self._inds_get(*layer_ix))
        ##print ("old_tags", old_tags)
        if iso and 'TREE' in old_tags:
            raise ValueError("You can't place isometric tensors above tree tensors.")
        rex = re.compile(self.site_tag_id.format(r"\d+", r"\d+"))
        old_tags = oset(filter(rex.match, old_tags))
        
        if not iso:
            old_tags.add('TREE')
            left_inds = bnds
        else:
            old_tags.add('ISO')
            left_inds = bnds
        
        TG = Tensor(G, inds=new_layer_ix + bnds, tags=tags | old_tags, left_inds=left_inds)
        
        self.reindex_(reindex_map)
        self |= TG
        return self
        



class TN3DUni(TensorNetwork3DVector,
              TensorNetwork3D,
              TensorNetwork):
        
    _EXTRA_PROPS = (
        '_site_tag_id',
        '_x_tag_id',
        '_y_tag_id',
        '_z_tag_id',
        '_site_ind_id',
        '_Lx',
        '_Ly',
        '_Lz'
    )
        
    @classmethod
    def empty(
        cls, 
        Lx, 
        Ly,
        Lz,
        phys_dim=2,
        data_type='float64',
        site_tag_id="I{},{},{}",
        site_ind_id="k{},{},{}",
        layer_ind_id="l{},{},{}",
    ):
        self = object.__new__(cls)
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz
        self._site_tag_id = site_tag_id
        self._site_ind_id = site_ind_id
        self._layer_ind_id = layer_ind_id
        self._x_tag_id = None
        self._y_tag_id = None
        self._z_tag_id = None
        
        
        ts = [
            Tensor(data=do("eye", phys_dim),
                   inds=[self.site_ind(i, j, k), 
                         self.layer_ind(i, j, k)],
                   tags=[self.site_tag(i, j, k), f"reg{i},{j},{k}", "const"],
                   left_inds=(self.site_ind(i, j, k),))
             for i, j, k in itertools.product(range(Lx), range(Ly), range(Lz))
        ]
        
        super().__init__(self, ts)


        return self
    
    @property
    def layer_ind_id(self):
        return self._layer_ind_id
    
    def layer_ind(self, i, j, k):
        return self._layer_ind_id.format(i, j, k)
    
    def reverse_gate(
        self,
        G,
        where,
        new_sites=None,
        tags=None,
        iso=True,
    ):
        """
        Parameters
        ----------
        G : array_like
        where : tuple[tuple[int]]
        new_sites : None or tuple[tuple[int]], optional
        tags : str or sequence of str, optional
        iso : bool, optional
        """
        tags = tags_to_oset(tags)
        
        if is_lone_coo(where):
            where = (where,)
        else:
            where = tuple(where)
            
        nbelow = len(where)
        layer_ix = [self.layer_ind(i, j, k) for i, j, k in where]
        bnds = [rand_uuid() for _ in range(nbelow)]
        reindex_map = dict(zip(layer_ix, bnds))
        ##print ("reindex_map", reindex_map)
        nabove = len(G.shape) - nbelow
        if new_sites is None:
            new_sites = where[:nabove]
        new_layer_ix = [self.layer_ind(i, j, k) for i, j, k in new_sites]
        


        #get tag of any tensor that has index "layer_ix"=[self.layer_ind(i, j) for i, j in where]
        old_tags = oset_union(t.tags for t in self._inds_get(*layer_ix))
        ##print ("old_tags", old_tags)
        if iso and 'TREE' in old_tags:
            raise ValueError("You can't place isometric tensors above tree tensors.")
        
        rex = re.compile(self.site_tag_id.format(r"\d+", r"\d+", r"\d+"))
        old_tags = oset(filter(rex.match, old_tags))
        
        if not iso:
            old_tags.add('TREE')
            left_inds = None
        else:
            old_tags.add('ISO')
            left_inds = bnds
        
        TG = Tensor(G, inds=new_layer_ix + bnds, tags=tags | old_tags, left_inds=left_inds)
        
        self.reindex_(reindex_map)
        self |= TG
        return self





def iso_Tn( tn, chi, size_Iso_x, size_Iso_y,list_tags,shared_I,list_scale,layer=None, L_x=None,L_y=None, 
           Iso=True, shift_x=0, shift_y=0,
           dis_x=1, dis_y=1,last_bond="off", cycle="on",data_type='float64', 
           seed_val=10,scale=0, label=0, dist_type="uniform"):

  shared_I.append(f"SI{scale},I{label}")
  list_scale.append(f"SI{scale}")


  lx_up=L_x//size_Iso_x if size_Iso_x != 0 else L_x 
  ly_up=L_y//size_Iso_y if size_Iso_y != 0 else L_y



  for i in range(shift_x, L_x,dis_x):
     for j in range(shift_y, L_y, dis_y):
        
        if cycle=="off":
                if (i+size_Iso_x-1)>L_x-1   or  (j+size_Iso_y-1)>L_y-1: break
        
        where=[]
        for i_x in range(size_Iso_x if size_Iso_x != 0 else 1):
         for j_y in range(size_Iso_y if size_Iso_y != 0 else 1):
              where.append( (  (i+i_x)%L_x,  (j+j_y)%L_y   )  )                 
                
             
     
        ##print ( i, j, where, [(f"{ ( shift_x+(   i//(size_Iso_x if size_Iso_x != 0 else 1) ) )%(lx_up) }",
            #f"{ ( shift_y+( j//(size_Iso_y if size_Iso_y != 0 else 1 )) )%(ly_up) }")]  ) 
        
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        if last_bond=="off":
         dims.insert(0, min(prod(dims), chi))
         ##print ("dim", dims)
         list_tags.append(f"I{layer},I{i},{j}")
         tn.reverse_gate(
            G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+i+j ), 
            where=where,
            iso=Iso,
            tags=["I", f"I{layer}",f"I{layer},I{i},{j}",f"SI{scale}",f"SI{scale},I{label}"],
            new_sites=[(f"{ ( shift_x+(   i//(size_Iso_x if size_Iso_x != 0 else 1) ) )%(lx_up) }",
            f"{ ( shift_y+( j//(size_Iso_y if size_Iso_y != 0 else 1 )) )%(ly_up) }")]
         )
        else: 
         list_tags.append(f"I{layer},I{i},{j}")
         tn.reverse_gate(
            G=qu.randn(dims,dtype=data_type,dist=dist_type, seed=seed_val+i+j),
            where=where,
            iso=Iso,
            tags=["I",f"I{layer}", f"I{layer},I{i},{j}",f"SI{scale}",f"SI{scale},I{label}"],
            )

  index_map={}
  for i in range((lx_up)):
   for j in range((ly_up)):
    index_map[f"l{i},{j}"] =f"l{ (i+lx_up-shift_x)%lx_up},{ (j+ly_up-shift_y)%ly_up}" 
    
  ##print (index_map)
  tn.reindex_(index_map)




def Uni_Tn( tn, chi, size_U_x, size_U_y,list_tags,shared_U,list_scale,layer=None, L_x=None,L_y=None, Iso_on=True, shift_x=0, shift_y=0,dis_x=1, dis_y=1,last_bond="off", cycle="on",data_type='float64', seed_val=10, scale=0,label=0, dist_type="uniform"):
  
  
  shared_U.append(f"SU{scale},U{label}")
  list_scale.append(f"SU{scale}")
  for i in range(shift_x, L_x,dis_x):
    for j in range(shift_y, L_y, dis_y):

        #print (i,j,shift_x,shift_y,L_x,L_y)
        where=[]
        if isinstance(size_U_x, list) or isinstance(size_U_y, list):

         for i_x,j_y in zip(size_U_x,size_U_y):
                if cycle=="off" and i+i_x<=L_x-1 and  j+j_y<=L_y-1: 
                    where.append( (  (i+i_x)%L_x,  (j+j_y)%L_y   )  )                 
                elif cycle=="on":
                     where.append( (  (i+i_x)%L_x,  (j+j_y)%L_y   )  )                 

        else:    
         for i_x in range(size_U_x):
          for j_y in range(size_U_y):
                #print (i_x, j_y, i+i_x<=L_x-1, j+j_y<=L_y-1, cycle=="off")
                if cycle=="off" and i+i_x<=L_x-1 and  j+j_y<=L_y-1: 
                    where.append( (  (i+i_x)%L_x,  (j+j_y)%L_y   )  )
                    #print ("where",where)
                elif cycle=="on":
                     where.append( (  (i+i_x)%L_x,  (j+j_y)%L_y   )  )                 

                
        #print (where)
        #take_out trivial unitary
        if len(where)==1: where = []
        #print (scale,where)
        
        
        if where:
          dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]*2
          if seed_val==0:
            G_v=qu.eye( int(prod(dims)**(1./2.)), dtype=data_type).reshape( dims )
            G_r=qu.randn(dims,dtype=data_type, dist=dist_type, seed=seed_val+i+j)
            G_v=G_v+G_r*(0.1)

          else:
            G_v=qu.randn(dims,dtype=data_type, dist=dist_type, seed=seed_val+i+j)

          list_tags.append(f"U{layer},I{i},{j}")
          tn.reverse_gate(G=G_v,where=where,iso=Iso_on,tags=["U",f"U{layer}",f"U{layer},I{i},{j}",f"SU{scale}",f"SU{scale},U{label}" ])




def Uni_Tn_3D( tn, chi, size_U_x, size_U_y,size_U_z,list_tags,list_scale,layer=None,
     L_x=None,L_y=None,L_z=None, Iso_on=True, shift_x=0, shift_y=0,shift_z=0,dis_x=1, 
     dis_y=1,dis_z=1,last_bond="off", cycle="on",data_type='float64', 
     dist_type="uniform",seed_val=10, scale=0, label=0):
    

  list_scale.append(f"SU{scale}")

  for i in range(shift_x, L_x,dis_x):
   for j in range(shift_y, L_y, dis_y):
    for k in range(shift_z, L_z, dis_z):
        where=[]
        if isinstance(size_U_x, list) or isinstance(size_U_y, list) or isinstance(size_U_z, list):


         for i_x, j_y, k_z in zip(size_U_x, size_U_y, size_U_z):
                if cycle=="off" and i+i_x<=L_x-1 and  j+j_y<=L_y-1 and k+k_z<=L_z-1: 
                    where.append( ( (i+i_x)%L_x, (j+j_y)%L_y, (k+k_z)%L_z)  )                 
                elif cycle=="on":
                     where.append( ( (i+i_x)%L_x, (j+j_y)%L_y, (k+k_z)%L_z)  )                 

        else:    
         for i_x in range(size_U_x):
          for j_y in range(size_U_y):
           for k_z in range(size_U_z):

                ##print (i_x, j_y, i+i_x<=L_x-1, j+j_y<=L_y-1, cycle=="off")
                if cycle=="off" and i+i_x<=L_x-1 and  j+j_y<=L_y-1 and k+k_z<=L_z-1: 
                    where.append( (  (i+i_x)%L_x,  (j+j_y)%L_y, (k+k_z)%L_z   )  )
                elif cycle=="on":
                     where.append( (  (i+i_x)%L_x,  (j+j_y)%L_y, (k+k_z)%L_z   )  )                 

                



        #print (i,j,k,where)

        #take_out trivial unitary
        if len(where)==1: where = []

        if where:
          dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]*2
          if seed_val==0:
            G_v=qu.eye( int(prod(dims)**(1./2.)), dtype=data_type).reshape( dims )
            G_r=qu.randn(dims,dtype=data_type, dist=dist_type, seed=seed_val+i+j)
            G_v=G_v+G_r*(0.1)
          else:
            G_v=qu.randn(dims,dtype=data_type, dist=dist_type, seed=seed_val+i+j)
          
          list_tags.append(f"U{layer},I{i},{j},{k}")
          tn.reverse_gate(G=G_v,where=where,iso=Iso_on,tags=["U",f"U{layer}",f"U{layer},I{i},{j},{k}",f"SU{scale}",f"SU{scale},U{label}"])




def iso_Tn_3D( tn, chi, size_Iso_x, size_Iso_y, size_Iso_z,list_tags,list_scale,layer=None, 
L_x=None,L_y=None, L_z=None, Iso=True, shift_x=0, shift_y=0,shift_z=0,dis_x=1, dis_y=1, dis_z=1,
last_bond="off", cycle="on",data_type='float64', seed_val=10, scale=0,dist_type="uniform",label=0):


  list_scale.append(f"SI{scale}")

  lx_up=L_x//size_Iso_x if size_Iso_x != 0 else L_x 
  ly_up=L_y//size_Iso_y if size_Iso_y != 0 else L_y
  lz_up=L_z//size_Iso_z if size_Iso_z != 0 else L_z


  ##print ("Hi","x",shift_x, L_x,dis_x, "y",shift_y, L_y, dis_y,"z" ,shift_z, L_z, dis_z)
  for i in range(shift_x, L_x,dis_x):
   for j in range(shift_y, L_y, dis_y):
    for k in range(shift_z, L_z, dis_z):

        ##print (i,j)
        if cycle=="off":
                ##print ((i+size_Iso_x-1)>L_x-1 , (j+size_Iso_y-1)>L_y-1)
                if (i+size_Iso_x-1)>L_x-1   or  (j+size_Iso_y-1)>L_y-1 or (k+size_Iso_z-1)>L_z-1: break
        
        where=[]
        for i_x in range(size_Iso_x if size_Iso_x != 0 else 1):
         for j_y in range(size_Iso_y if size_Iso_y != 0 else 1):
          for k_z in range(size_Iso_z if size_Iso_z != 0 else 1):
              where.append( (  (i+i_x)%L_x,  (j+j_y)%L_y, (k+k_z)%L_z   )  )                 
                
             
     
        
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        if last_bond=="off":
         ##print ( i, j,k, where, [(f"{ ( shift_x+(   i//(size_Iso_x if size_Iso_x != 0 else 1) ) )%(lx_up) }",
          #  f"{ ( shift_y+( j//(size_Iso_y if size_Iso_y != 0 else 1 )) )%(ly_up) }",f"{ ( shift_z+( k//(size_Iso_z if size_Iso_z != 0 else 1 )) )%(lz_up) }")]  )
         dims.insert(0, min(prod(dims), chi))
         ##print ("dim", dims)
         list_tags.append(f"I{layer},I{i},{j},{k}")
         tn.reverse_gate(
            G=qu.randn(dims,dtype=data_type, dist=dist_type, seed=seed_val+i+j ), 
            where=where,
            iso=Iso,
            tags=["I", f"I{layer}",f"I{layer},I{i},{j},{k}",f"SI{scale}",f"SI{scale},I{label}"],
            new_sites=[(f"{ ( shift_x+(   i//(size_Iso_x if size_Iso_x != 0 else 1) ) )%(lx_up) }",
            f"{ ( shift_y+( j//(size_Iso_y if size_Iso_y != 0 else 1 )) )%(ly_up) }", f"{ ( shift_z+( k//(size_Iso_z if size_Iso_z != 0 else 1 )) )%(lz_up) }")]
         )
        else: 
         list_tags.append(f"I{layer},I{i},{j},{k}")
         tn.reverse_gate(
            G=qu.randn(dims,dtype=data_type, dist=dist_type, seed=seed_val+i+j), 
            where=where,
            iso=Iso,
            tags=["I",f"I{layer}",f"I{layer},I{i},{j},{k}",f"SI{scale}",f"SI{scale},I{label}"],
            )

  index_map={}
  for i in range((lx_up)):
   for j in range((ly_up)):
    for k in range((lz_up)):

     index_map[f"l{i},{j},{k}"] =f"l{ (i+lx_up-shift_x)%lx_up},{ (j+ly_up-shift_y)%ly_up},{ (k+lz_up-shift_z)%lz_up}" 
    
  ##print (index_map)
  tn.reindex_(index_map)




def Heis_local_Ham_open_tree(N_x,N_y,data_type="float64"):
 Z = qu.pauli('Z',dtype=data_type) * (0.5)
 X = qu.pauli('X',dtype=data_type) * (0.5)
 Y=np.array([[0, -1],[1,0]]) * (0.5)


 X=X.astype(data_type)
 Z=Z.astype(data_type)
 Y=Y.astype(data_type)


    
 list_sites=[]
 list_inter=[]
 ##print (qu.ham_heis(2))



 for i in range(N_x-1): 
  for j in range(N_y): 
   list_sites.append( [ ( i,j), ((i+1),j) ]   )
   

 for i in range(N_x): 
  for j in range(N_y-1): 
   list_sites.append( [ ( i,j), (i, (j+1)) ]   )


 list_inter=[X,Z,Y]
 return list_sites, list_inter




















def Heis_local_Ham_open(N_x,N_y,data_type="float64",phys_dim=2):

 list_sites=[]
 list_inter=[]
 ##print (qu.ham_heis(2))
 #H=qu.ham_heis(2).astype(data_type)
 if phys_dim==2:
    h_h=qu.ham_heis(2).astype(data_type)
 else:
    h_h=np.eye(phys_dim*phys_dim).astype(data_type)
    h_h=qu.randn((phys_dim, phys_dim, phys_dim, phys_dim),dtype=data_type, dist="uniform", seed=10)



 for i in range(N_x-1): 
  for j in range(N_y): 
   list_sites.append( [ ( i,j), ((i+1),j) ]   )
   list_inter.append( h_h )

 for i in range(N_x): 
  for j in range(N_y-1): 
   list_sites.append( [ ( i,j), (i, (j+1)) ]   )
   list_inter.append( h_h )

 return list_sites, list_inter





def Heis_local_Ham_open_long(N_x,N_y,data_type="float64",phys_dim=2):

 list_sites=[]
 list_inter=[]
 ##print (qu.ham_heis(2))
 #H=qu.ham_heis(2).astype(data_type)
 if phys_dim==2:
    h_h=qu.ham_heis(2).astype(data_type)
 else:
    h_h=np.eye(phys_dim*phys_dim).astype(data_type)
    h_h=qu.randn((phys_dim, phys_dim, phys_dim, phys_dim),dtype=data_type, dist="uniform", seed=10)


 val=0
 for i, j in itertools.product(range(N_x), range(N_y)):
    for i1, j1 in itertools.product(range(N_x), range(N_y)):
  
      if i != i1   or j != j1:
       ii=i*N_y+j
       ii_=i1*N_y+j1
       if ii_<ii:
         val+=1
         r=(abs(i-i1)**2+abs(j-j1)**2)**(0.5)
         #print  ( (i,j),(i1,j1),r,val   )
         list_sites.append( [ ( i,j), (i1,j1) ]   )
         list_inter.append( h_h/r )


 return list_sites, list_inter

















def Heis_local_Ham_cycle(N_x,N_y):

 list_sites=[]
 list_inter=[]
 ##print (qu.ham_heis(2))

 for i in range(N_x): 
  for j in range(N_y): 
   list_sites.append( [ ( i,j), ((i+1)%N_x,j) ]   )
   list_inter.append( qu.ham_heis(2) )

 for i in range(N_x): 
  for j in range(N_y): 
   list_sites.append( [ ( i,j), (i, (j+1)%N_y) ]   )
   list_inter.append( qu.ham_heis(2) )

 return list_sites, list_inter


def  mpo_2d_His(L_x,L_y,L_L, data_type="float64"):

 Z=qu.pauli('Z',dtype=data_type) * 0.5
 X=qu.pauli('X',dtype=data_type) * 0.5
 Y= np.array([[0, -1],[1,0]]) * 0.5
 I=qu.pauli('I',dtype=data_type)
 Y=Y.astype(data_type)
 X=X.astype(data_type)
 Z=Z.astype(data_type)

 Ham=[X, Y, Z]
 MPO_I=MPO_identity(L_L, phys_dim=2)
 MPO_result=MPO_identity(L_L, phys_dim=2)
 MPO_result=MPO_result*0.0
 MPO_f=MPO_result*0.0
 
 
 max_bond_val=300
 cutoff_val=1.0e-12
 for count, elem in enumerate (Ham):
   for i, j in itertools.product(range(L_x-1), range(L_y)):
      ii=i*L_y+j
      ii_=(i+1)*L_y+j
      Wl = np.zeros([ 1, 2, 2], dtype=data_type)
      W = np.zeros([1, 1, 2, 2], dtype=data_type)
      Wr = np.zeros([ 1, 2, 2], dtype=data_type)
   
      Wl[ 0,:,:]=elem
      W[ 0,0,:,:]=elem
      Wr[ 0,:,:]=elem
      W_list=[Wl]+[W]*(L_L-2)+[Wr]
      MPO_I=MPO_identity(L_L, phys_dim=2 )
      if count!=1:
         MPO_I[ii].modify(data=W_list[ii])
         MPO_I[ii_].modify(data=W_list[ii_])
         MPO_result=MPO_result+MPO_I
         MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )
      elif count==1:
         MPO_I[ii].modify(data=W_list[ii])
         MPO_I[ii_].modify(data=W_list[ii_]*-1.0)
         MPO_result=MPO_result+MPO_I
         MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )



 for count, elem in enumerate (Ham):
   for i, j in itertools.product(range(L_x), range(L_y-1)):
      ii=i*L_y+j
      ii_=i*L_y+j+1
      Wl = np.zeros([ 1, 2, 2], dtype=data_type)
      W = np.zeros([1, 1, 2, 2], dtype=data_type)
      Wr = np.zeros([ 1, 2, 2], dtype=data_type)

      Wl[ 0,:,:]=elem
      W[ 0,0,:,:]=elem
      Wr[ 0,:,:]=elem
      W_list=[Wl]+[W]*(L_L-2)+[Wr]

      MPO_I=MPO_identity(L_L, phys_dim=2 )
      if count!=1:
         MPO_I[ii].modify(data=W_list[ii])
         MPO_I[ii_].modify(data=W_list[ii_])
         MPO_result=MPO_result+MPO_I
         MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )
      elif count==1:
         MPO_I[ii].modify(data=W_list[ii])
         MPO_I[ii_].modify(data=W_list[ii_]*-1.)
         MPO_result=MPO_result+MPO_I
         MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )



 MPO_f=MPO_result
 MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )
 print ( MPO_f.show() )
 return  MPO_f 













def  mpo_2d_His_long(L_x,L_y,L_L, data_type="float64",cutoff_val=1.0e-12):

 Z=qu.pauli('Z',dtype=data_type) * 0.5
 X=qu.pauli('X',dtype=data_type) * 0.5
 Y= np.array([[0, -1],[1,0]]) * 0.5
 I=qu.pauli('I',dtype=data_type)
 Y=Y.astype(data_type)
 X=X.astype(data_type)
 Z=Z.astype(data_type)

 Ham=[X, Y, Z]
 MPO_I=MPO_identity(L_L, phys_dim=2)
 MPO_result=MPO_identity(L_L, phys_dim=2)
 MPO_result=MPO_result*0.0
 MPO_f=MPO_result*0.0
 
 
 max_bond_val=300
 cutoff_val=1.0e-12
 for count, elem in enumerate (Ham):
   for i, j in itertools.product(range(L_x), range(L_y)):
    for i1, j1 in itertools.product(range(L_x), range(L_y)):
     if i !=i1 or j!=j1:
      r=(abs(i-i1)**2+abs(j-j1)**2)**(0.5)
      ii=i*L_y+j
      ii_=i1*L_y+j1
      if ii_<ii:
         Wl = np.zeros([ 1, 2, 2], dtype=data_type)
         W = np.zeros([1, 1, 2, 2], dtype=data_type)
         Wr = np.zeros([ 1, 2, 2], dtype=data_type)
      
         Wl[ 0,:,:]=elem
         W[ 0,0,:,:]=elem
         Wr[ 0,:,:]=elem
         W_list=[Wl]+[W]*(L_L-2)+[Wr]
         MPO_I=MPO_identity(L_L, phys_dim=2 )
         if count!=1:
            MPO_I[ii].modify(data=W_list[ii])
            MPO_I[ii_].modify(data=W_list[ii_]*(r**-1))
            MPO_result=MPO_result+MPO_I
            MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )
         elif count==1:
            MPO_I[ii].modify(data=W_list[ii])
            MPO_I[ii_].modify(data=W_list[ii_]*-1.0*(r**-1))
            MPO_result=MPO_result+MPO_I
            MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 MPO_f=MPO_result
 MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )
 print ( MPO_f.show() )
 return  MPO_f 


def  mpo_3d_His(L_x,L_y,L_z,L_L, data_type="float64", chi=200,cutoff_val=1.0e-12):

 Z=qu.pauli('Z',dtype=data_type) * 0.5
 X=qu.pauli('X',dtype=data_type) * 0.5
 Y= np.array([[0, -1],[1,0]]) * 0.5
 I=qu.pauli('I',dtype=data_type)
 Y=Y.astype(data_type)
 X=X.astype(data_type)
 Z=Z.astype(data_type)

 Ham=[X, Y, Z]
 MPO_I=MPO_identity(L_L, phys_dim=2)
 MPO_result=MPO_identity(L_L, phys_dim=2)
 MPO_result=MPO_result*0.0
 MPO_f=MPO_result*0.0
 
 
 max_bond_val=chi
 cutoff_val=cutoff_val
 for count, elem in enumerate (Ham):
   for i, j, k in itertools.product(range(L_x-1), range(L_y),range(L_z)):
      print ("1", i,j,k,MPO_result.max_bond())
      ii=i*L_y*L_z+j*L_z+k
      ii_=(i+1)*L_y*L_z+j*L_z+k
      Wl = np.zeros([ 1, 2, 2], dtype=data_type)
      W = np.zeros([1, 1, 2, 2], dtype=data_type)
      Wr = np.zeros([ 1, 2, 2], dtype=data_type)
   
      Wl[ 0,:,:]=elem
      W[ 0,0,:,:]=elem
      Wr[ 0,:,:]=elem
      W_list=[Wl]+[W]*(L_L-2)+[Wr]
      MPO_I=MPO_identity(L_L, phys_dim=2 )
      if count!=1:
         MPO_I[ii].modify(data=W_list[ii])
         MPO_I[ii_].modify(data=W_list[ii_])
         MPO_result=MPO_result+MPO_I
         MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )
      elif count==1:
         MPO_I[ii].modify(data=W_list[ii])
         MPO_I[ii_].modify(data=W_list[ii_]*-1.0)
         MPO_result=MPO_result+MPO_I
         MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 for count, elem in enumerate (Ham):
   for i, j, k in itertools.product(range(L_x), range(L_y-1),range(L_z)):
      print ("2", i,j,k,MPO_result.max_bond())
      ii=i*L_y*L_z+j*L_z+k
      ii_=i*L_y*L_z+(j+1)*L_z+k
      Wl = np.zeros([ 1, 2, 2], dtype=data_type)
      W = np.zeros([1, 1, 2, 2], dtype=data_type)
      Wr = np.zeros([ 1, 2, 2], dtype=data_type)

      Wl[ 0,:,:]=elem
      W[ 0,0,:,:]=elem
      Wr[ 0,:,:]=elem
      W_list=[Wl]+[W]*(L_L-2)+[Wr]

      MPO_I=MPO_identity(L_L, phys_dim=2 )
      if count!=1:
         MPO_I[ii].modify(data=W_list[ii])
         MPO_I[ii_].modify(data=W_list[ii_])
         MPO_result=MPO_result+MPO_I
         MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )
      elif count==1:
         MPO_I[ii].modify(data=W_list[ii])
         MPO_I[ii_].modify(data=W_list[ii_]*-1.)
         MPO_result=MPO_result+MPO_I
         MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 for count, elem in enumerate (Ham):
   for i, j, k in itertools.product(range(L_x), range(L_y),range(L_z-1)):
      print ("3", i,j,k, MPO_result.max_bond() )
      ii=i*L_y*L_z+j*L_z+k
      ii_=i*L_y*L_z+j*L_z+k+1
      Wl = np.zeros([ 1, 2, 2], dtype=data_type)
      W = np.zeros([1, 1, 2, 2], dtype=data_type)
      Wr = np.zeros([ 1, 2, 2], dtype=data_type)

      Wl[ 0,:,:]=elem
      W[ 0,0,:,:]=elem
      Wr[ 0,:,:]=elem
      W_list=[Wl]+[W]*(L_L-2)+[Wr]

      MPO_I=MPO_identity(L_L, phys_dim=2 )
      if count!=1:
         MPO_I[ii].modify(data=W_list[ii])
         MPO_I[ii_].modify(data=W_list[ii_])
         MPO_result=MPO_result+MPO_I
         MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )
      elif count==1:
         MPO_I[ii].modify(data=W_list[ii])
         MPO_I[ii_].modify(data=W_list[ii_]*-1.)
         MPO_result=MPO_result+MPO_I
         MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


 MPO_f=MPO_result
 MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )
 print ( MPO_f.show() )
 return  MPO_f 




def Heis_local_Ham_open_3D(N_x,N_y,N_z, data_type="float64", phys_dim=2):

 list_sites=[]
 list_inter=[]
 ##print (qu.ham_heis(2))
 if phys_dim==2:
   h_h=qu.ham_heis(2).astype(data_type)
 else:
   h_h=np.eye(phys_dim*phys_dim).astype(data_type)
   h_h=qu.randn((phys_dim, phys_dim, phys_dim, phys_dim),dtype=data_type, dist="uniform", seed=10)


 for i in range(N_x-1): 
  for j in range(N_y): 
   for k in range(N_z): 

    list_sites.append( [ ( i,j,k), (i+1,j,k) ]   )
    list_inter.append( h_h )

 for i in range(N_x): 
  for j in range(N_y-1): 
   for k in range(N_z): 
    list_sites.append( [ ( i,j,k), (i,j+1,k) ]   )
    list_inter.append( h_h )


 for i in range(N_x): 
  for j in range(N_y): 
   for k in range(N_z-1): 
    list_sites.append( [ ( i,j,k), (i,j,k+1) ]   )
    list_inter.append( h_h )


 return list_sites, list_inter







def Heis_local_Ham_cycle_3D(N_x,N_y,N_z, data_type="float64", phys_dim=2):

 list_sites=[]
 list_inter=[]
 ##print (qu.ham_heis(2))
 if phys_dim==2:
   h_h=qu.ham_heis(2).astype(data_type)
 else:
   h_h=np.eye(phys_dim*phys_dim).astype(data_type)
   h_h=qu.randn((phys_dim, phys_dim, phys_dim, phys_dim),dtype=data_type, dist="uniform", seed=10)


 for i in range(N_x): 
  for j in range(N_y): 
   for k in range(N_z): 

    list_sites.append( [ ( i,j,k), ( (i+1)%N_x ,j,k) ]   )
    list_inter.append( h_h )

 for i in range(N_x): 
  for j in range(N_y): 
   for k in range(N_z): 
    list_sites.append( [ ( i,j,k), (i,(j+1)%N_y,k) ]   )
    list_inter.append( h_h )


 for i in range(N_x): 
  for j in range(N_y): 
   for k in range(N_z): 
    list_sites.append( [ ( i,j,k), (i,j,(k+1)%N_z ) ]   )
    list_inter.append( h_h )


 return list_sites, list_inter







def Heis_local_Ham_cycle_3d(N_x,N_y, N_z):

 list_sites=[]
 list_inter=[]
 ##print (qu.ham_heis(2))

 for i in range(N_x): 
  for j in range(N_y): 
   for k in range(N_z): 
    list_sites.append( [ ( i,j,k), ((i+1)%N_x,j,k) ]   )
    list_inter.append( qu.ham_heis(2) )

 for i in range(N_x): 
  for j in range(N_y): 
   for k in range(N_z): 
     list_sites.append( [ ( i,j,k), (i, (j+1)%N_y,k) ]   )
     list_inter.append( qu.ham_heis(2) )

 for i in range(N_x): 
  for j in range(N_y): 
   for k in range(N_z): 
     list_sites.append( [ ( i,j,k), (i, j, (k+1)%N_z) ]   )
     list_inter.append( qu.ham_heis(2) )


 return list_sites, list_inter


def uni_xy_44_mps(tn_U, N_x_l, N_y_l, chi, num_layer, list_tags_U,label_listU, shared_U,lsit_scale,scale=0, seed_val=10, cycle="off", data_type="float64",dist_type="uniform"):


   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, 
   shift_x=0, shift_y=0,
   dis_x=2, dis_y=2, 
   cycle=cycle,seed_val=seed_val, 
   data_type=data_type,scale=scale, label=label_listU[0],dist_type=dist_type)



   num_layer[0]+=1
   label_listU[0]+=1
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0],
    L_x=N_x_l, L_y=N_y_l, Iso_on=True, 
    shift_x=1, shift_y=1,
    dis_x=2, dis_y=2, 
cycle=cycle,seed_val=seed_val,data_type=data_type,scale=scale,label=label_listU[0],
    dist_type=dist_type)















def uni_xy_44(tn_U, N_x_l, N_y_l, chi, num_layer, list_tags_U,label_listU, shared_U,lsit_scale,scale=0, seed_val=10, cycle="off", data_type="float64",dist_type="uniform"):


   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, 
   shift_x=3, shift_y=1,
   dis_x=4, dis_y=4, 
   cycle=cycle,seed_val=seed_val, 
   data_type=data_type,scale=scale, label=label_listU[0],dist_type=dist_type)


   num_layer[0]+=1
   label_listU[0]+=1
   size_U_x=1
   size_U_y=2
   Uni_Tn( tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, 
          shift_x=1, shift_y=3,dis_x=4, dis_y=4, cycle=cycle,seed_val=seed_val,data_type=data_type,scale=scale,label=label_listU[0],dist_type=dist_type)

   num_layer[0]+=1
   label_listU[0]+=1
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0],
    L_x=N_x_l, L_y=N_y_l, Iso_on=True, 
    shift_x=3, shift_y=2,
    dis_x=4, dis_y=4, 
    cycle=cycle,seed_val=seed_val,data_type=data_type,scale=scale,label=label_listU[0],
    dist_type=dist_type)

   num_layer[0]+=1
   label_listU[0]+=1
   size_U_x=1
   size_U_y=2
   Uni_Tn( tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, 
          shift_x=2, shift_y=3,dis_x=4, dis_y=4, cycle=cycle,seed_val=seed_val,data_type=data_type,scale=scale,label=label_listU[0],dist_type=dist_type)
   num_layer[0]+=1
   label_listU[0]+=1





def uni_xy_44_full(tn_U, N_x_l, N_y_l, chi, num_layer, list_tags_U,label_listU, shared_U,lsit_scale,scale=0, seed_val=10, cycle="off", data_type="float64",dist_type="uniform"):

   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, shift_x=3, shift_y=0,dis_x=4, dis_y=1, cycle=cycle,seed_val=seed_val, data_type=data_type,scale=scale, label=label_listU[0],dist_type=dist_type)


   num_layer[0]+=1
   label_listU[0]+=1
   size_U_x=1
   size_U_y=2
   Uni_Tn( tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, 
          shift_x=0, shift_y=3,dis_x=1, dis_y=4, cycle=cycle,seed_val=seed_val,data_type=data_type,scale=scale,label=label_listU[0],dist_type=dist_type)
   num_layer[0]+=1
   label_listU[0]+=1



def uni_xy_44_Iso(tn_U, N_x_l, N_y_l, chi, num_layer, list_tags_U,label_listU, shared_U,lsit_scale,scale=0, seed_val=10, cycle="off", data_type="float64",dist_type="uniform"):
   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, 
   shift_x=1, shift_y=1,
   dis_x=4, dis_y=4, 
   cycle=cycle,
   seed_val=seed_val, 
   data_type=data_type,scale=scale, 
   label=label_listU[0],dist_type=dist_type)


   num_layer[0]+=1
   label_listU[0]+=1
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, 
   shift_x=1, shift_y=2,
   dis_x=4, dis_y=4, 
   cycle=cycle,seed_val=seed_val, 
   data_type=data_type,scale=scale, 
   label=label_listU[0],dist_type=dist_type)
   num_layer[0]+=1
   label_listU[0]+=1



def uni_xy_22_Iso(tn_U, N_x_l, N_y_l, chi, num_layer, list_tags_U,label_listU, shared_U,lsit_scale,scale=0, seed_val=10, cycle="off", data_type="float64",dist_type="uniform"):

   #print ("l,l",N_x_l, N_y_l)
   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, shift_x=0, shift_y=0,dis_x=2, dis_y=2, cycle=cycle,seed_val=seed_val, data_type=data_type,scale=scale, label=label_listU[0],dist_type=dist_type)


   num_layer[0]+=1
   label_listU[0]+=1
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, shift_x=0, shift_y=1,dis_x=2, dis_y=2, cycle=cycle,seed_val=seed_val, data_type=data_type,scale=scale, label=label_listU[0],dist_type=dist_type)
   num_layer[0]+=1
   label_listU[0]+=1










def uni_dense_22(tn_U, N_x_l, N_y_l, chi,num_layer,list_tags_U, label_listU,shared_U,lsit_scale,seed_val=10, cycle="off", data_type="float64", scale=0):
   num_layer[0]+=1
   size_U_x=2
   size_U_y=2
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, shift_x=2, shift_y=2,dis_x=3, dis_y=3,cycle=cycle,seed_val=seed_val,data_type=data_type,scale=scale, label=label_listU[0])
   num_layer[0]+=1
   label_listU[0]+=1



def uni_local_x_2(tn_U, N_x_l, N_y_l, layer_num,chi, seed_val=10, cycle="off", dis_opt=2,shift_x=0,shift_y=1):
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, layer=layer_num, L_x=N_x_l, L_y=N_y_l, Iso_on=True, shift_x=shift_x, shift_y=shift_y,dis_x=dis_opt, dis_y=1, cycle=cycle,seed_val=seed_val)


def uni_local_y_2(tn_U, N_x_l, N_y_l, layer_num,chi, seed_val=10, cycle="off",dis_opt=2, shift_x=0,shift_y=1):
   size_U_x=1
   size_U_y=2
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, layer=layer_num, L_x=N_x_l, L_y=N_y_l, Iso_on=True, shift_x=shift_x, shift_y=shift_y,dis_x=1, dis_y=dis_opt, cycle=cycle,seed_val=seed_val)






def uni_xy_33(tn_U, N_x_l, N_y_l, chi,num_layer,list_tags_U, label_listU,shared_U,lsit_scale,seed_val=10, cycle="off", data_type="float64", scale=0, dist_type="uniform"):

   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, shift_x=2, shift_y=1,dis_x=3, dis_y=3, cycle=cycle,seed_val=seed_val,data_type=data_type,scale=scale, label=label_listU[0],dist_type=dist_type)

   num_layer[0]+=1
   label_listU[0]+=1
   size_U_x=1
   size_U_y=2
   Uni_Tn( tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True,shift_x=1, shift_y=2,dis_x=3, dis_y=3, cycle=cycle,seed_val=seed_val,data_type=data_type,scale=scale,label=label_listU[0],dist_type=dist_type)
   num_layer[0]+=1
   label_listU[0]+=1





def uni_xy_33_full(tn_U, N_x_l, N_y_l, chi,num_layer,list_tags_U, label_listU,shared_U,lsit_scale,seed_val=10, cycle="off", data_type="float64", scale=0):
   #print (scale)
   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   Uni_Tn(tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True, shift_x=2, shift_y=0,dis_x=3, dis_y=1, cycle=cycle,seed_val=seed_val,data_type=data_type,scale=scale, label=label_listU[0])

   num_layer[0]+=1
   label_listU[0]+=1
   size_U_x=1
   size_U_y=2
   Uni_Tn( tn_U, chi, size_U_x, size_U_y, list_tags_U,shared_U,lsit_scale,layer=num_layer[0], L_x=N_x_l, L_y=N_y_l, Iso_on=True,shift_x=0, shift_y=2,dis_x=1, dis_y=3, cycle=cycle,seed_val=seed_val,data_type=data_type,scale=scale,label=label_listU[0])
   num_layer[0]+=1
   label_listU[0]+=1




def uni_diag_33(tn_U, N_x_l, N_y_l,layer_num,chi,seed_val=10, cycle="off"):
   #diagonal-direction_x
   size_U_x=[0,-1]
   size_U_y=[0,1]
   Uni_Tn( tn_U, chi, size_U_x, size_U_y, layer=4*layer_num+2, L_x=N_x_l, L_y=N_y_l, Iso_on=True, 
          shift_x=3, shift_y=2,
      dis_x=3, dis_y=3, cycle=cycle,seed_val=seed_val)

   #diagonal-direction_y
   size_U_x=[0,1]
   size_U_y=[0,1]
   Uni_Tn( tn_U, chi, size_U_x, size_U_y, layer=4*layer_num+3, L_x=N_x_l, L_y=N_y_l, Iso_on=True, 
          shift_x=2, shift_y=2,
      dis_x=3, dis_y=3, cycle=cycle,seed_val=seed_val)







def  Iso_33(tn_U, N_x_l, N_y_l,layer_num,chi,seed_val=10,last_bond="off"):
  size_Iso_x=0
  size_Iso_y=3
  l_x=N_x_l
  l_y=N_y_l
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,layer=layer_num[0], L_x=l_x,L_y=l_y, Iso_on=True,shift_x=0, shift_y=0,
         dis_x=1, dis_y=3, cycle="on",seed_val=seed_val)
  layer_num[0]+=1
  size_Iso_x=3
  size_Iso_y=0
  l_x=N_x_l
  l_y=N_y_l//3
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,layer=layer_num[0], L_x=l_x,L_y=l_y, Iso_on=True,shift_x=0, shift_y=0,
          dis_x=3, dis_y=1, last_bond=last_bond, cycle="on",seed_val=seed_val)
  layer_num[0]+=1




def uni_xy_44_3D_iso_corner(tn_U, N_x_l, N_y_l, N_z_l,chi,num_layer,list_tags_U, list_scale, scale,seed_val=10, cycle="off", dist_type="uniform"):


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=0,shift_z=0,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=0,shift_z=0,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=0,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=0,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

#####################################################################################################################################
   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=2,shift_z=0,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, 
   scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=2,shift_z=0,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, 
   scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=2,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, 
   scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=2,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,
   dist_type=dist_type)







def uni_xy_44_3D_iso_corner_full(tn_U, N_x_l, N_y_l, N_z_l,chi,num_layer,list_tags_U, list_scale, scale,seed_val=10, cycle="off", dist_type="uniform"):


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=0,shift_z=0,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=0,shift_z=0,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=0,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=0,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

###################more corner#################
   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=0,shift_z=1,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=0,shift_z=2,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=0,shift_z=1,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=0,shift_z=2,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)




   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=0,shift_z=0,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=2, shift_y=0,shift_z=0,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)



   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=0,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=2, shift_y=0,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)







#####################################################################################################################################
   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=2,shift_z=0,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, 
   scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=2,shift_z=0,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, 
   scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=2,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, 
   scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=2,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,
   dist_type=dist_type)




###################more corner#################
   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=2,shift_z=1,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=0, shift_y=2,shift_z=2,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=2,shift_z=1,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=2,shift_z=2,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)




   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=2,shift_z=0,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=2, shift_y=2,shift_z=0,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)



   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=2,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=2, shift_y=2,shift_z=3,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)















def uni_xy_44_3D_iso_center_zx(tn_U, N_x_l, N_y_l, N_z_l,chi,num_layer,list_tags_U, list_scale, scale,seed_val=10, cycle="off", dist_type="uniform"):

   num_layer[0]+=1
############y-direction#############
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=0, shift_z=1,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   num_layer[0]+=1
############y-direction#############
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=0, shift_z=2,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

#####################################################################################################################

   num_layer[0]+=1
############y-direction#############
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=1, shift_z=1,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   num_layer[0]+=1
############y-direction#############
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=1, shift_z=2,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

##########################################################################################################################


   num_layer[0]+=1
############y-direction#############
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=2, shift_z=1,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   num_layer[0]+=1
############y-direction#############
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=2, shift_z=2,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)
################################################################################################################################



   num_layer[0]+=1
############y-direction#############
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=3, shift_z=1,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   num_layer[0]+=1
############y-direction#############
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=3, shift_z=2,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)
################################################################################################################################


def uni_xy_44_3D_iso_center_zy(tn_U, N_x_l, N_y_l, N_z_l,chi,num_layer,list_tags_U, list_scale, scale,seed_val=10, cycle="off", dist_type="uniform"):

#    num_layer[0]+=1
# ############y-direction#############
#    size_U_x=1
#    size_U_y=2
#    size_U_z=1
#    Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
#    L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
#    shift_x=1, shift_y=0, shift_z=2,dis_x=4,dis_y=4,dis_z=4,
#    cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

#    num_layer[0]+=1
# ############y-direction#############
#    size_U_x=1
#    size_U_y=2
#    size_U_z=1
#    Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
#    L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
#    shift_x=2, shift_y=0, shift_z=1,dis_x=4,dis_y=4,dis_z=4,
#    cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)



   num_layer[0]+=1
############y-direction#############
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=1, shift_z=1,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

#    num_layer[0]+=1
# ############y-direction#############
#    size_U_x=1
#    size_U_y=2
#    size_U_z=1
#    Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
#    L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
#    shift_x=1, shift_y=1, shift_z=2,dis_x=4,dis_y=4,dis_z=4,
#    cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

#    num_layer[0]+=1
# ############y-direction#############
#    size_U_x=1
#    size_U_y=2
#    size_U_z=1
#    Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
#    L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
#    shift_x=2, shift_y=1, shift_z=1,dis_x=4,dis_y=4,dis_z=4,
#    cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

#    num_layer[0]+=1
# ############y-direction#############
#    size_U_x=1
#    size_U_y=2
#    size_U_z=1
#    Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
#    L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
#    shift_x=2, shift_y=1, shift_z=2,dis_x=4,dis_y=4,dis_z=4,
#    cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)




#    num_layer[0]+=1
# ############y-direction#############
#    size_U_x=1
#    size_U_y=2
#    size_U_z=1
#    Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
#    L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
#    shift_x=1, shift_y=2, shift_z=2,dis_x=4,dis_y=4,dis_z=4,
#    cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

#    num_layer[0]+=1
# ############y-direction#############
#    size_U_x=1
#    size_U_y=2
#    size_U_z=1
#    Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
#    L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
#    shift_x=2, shift_y=2, shift_z=1,dis_x=4,dis_y=4,dis_z=4,
#    cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)














def uni_xy_44_3D(tn_U, N_x_l, N_y_l, N_z_l,chi,num_layer,list_tags_U, list_scale, scale,seed_val=10, cycle="off", dist_type="uniform"):

   num_layer[0]+=1
############y-direction#############
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=3,shift_z=1,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=2, shift_y=3,shift_z=1,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)

   num_layer[0]+=1

   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=3,shift_z=2,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)

   num_layer[0]+=1

   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=2, shift_y=3,shift_z=2,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)


############x-direction#############
   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=1,shift_z=1,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=2,shift_z=1,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=1,shift_z=2,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=2,shift_z=2,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)

############z-direction#############
   num_layer[0]+=1
   size_U_x=1
   size_U_y=1
   size_U_z=2
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=1,shift_z=3,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=1
   size_U_z=2
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=2,shift_z=3,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=1
   size_U_z=2
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=2, shift_y=1,shift_z=3,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=1
   size_U_z=2
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=2, shift_y=2,shift_z=3,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)
   num_layer[0]+=1









def uni_xy_44_3D_full(tn_U, N_x_l, N_y_l, N_z_l,chi,num_layer,list_tags_U, list_scale, scale,seed_val=10, cycle="off", dist_type="uniform"):


   for i_iter in range(4):
     for k_iter in range(4):
           num_layer[0]+=1
           size_U_x=1
           size_U_y=2
           size_U_z=1
           Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
                              L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
                              shift_x=i_iter, shift_y=3,shift_z=k_iter,dis_x=4,dis_y=4,dis_z=4,
                              cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   for j_iter in range(4):
     for k_iter in range(4):
           num_layer[0]+=1
           size_U_x=2
           size_U_y=1
           size_U_z=1
           Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
                              L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
                              shift_x=3, shift_y=j_iter,shift_z=k_iter,dis_x=4,dis_y=4,dis_z=4,
                              cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   for i_iter in range(4):
    for j_iter in range(4):
           num_layer[0]+=1
           size_U_x=1
           size_U_y=1
           size_U_z=2
           Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
                              L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
                              shift_x=i_iter, shift_y=j_iter,shift_z=3,dis_x=4,dis_y=4,dis_z=4,
                              cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)











def uni_xy_44_3D_sparse(tn_U, N_x_l, N_y_l, N_z_l,chi,num_layer,list_tags_U, list_scale, scale,seed_val=10, cycle="off", dist_type="uniform"):

   num_layer[0]+=1
############y-direction#############
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=3,shift_z=1,
   dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   num_layer[0]+=1

   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=2, shift_y=3,shift_z=2,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)




############x-direction#############
   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=1,shift_z=1,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)



   num_layer[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=3, shift_y=2,shift_z=2,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)

############z-direction#############


   num_layer[0]+=1
   size_U_x=1
   size_U_y=1
   size_U_z=2
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=1, shift_y=1,shift_z=3,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)

   num_layer[0]+=1
   size_U_x=1
   size_U_y=1
   size_U_z=2
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z, list_tags_U,list_scale,layer=num_layer[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, Iso_on=True, 
   shift_x=2, shift_y=2,shift_z=3,dis_x=4,dis_y=4,dis_z=4,
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)








def uni_xy_22_3D_full(tn_U, N_x_l, N_y_l,N_z_l, layer_num,list_tags,chi, list_scale,scale,seed_val=10,
 cycle="off",dist_type="uniform"):

   layer_num[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=0, shift_y=0, shift_z=0,dis_x=2, dis_y=2,dis_z=2, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   layer_num[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=0, shift_y=0, shift_z=1,dis_x=2, dis_y=2,dis_z=2, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   layer_num[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=0, shift_y=1, shift_z=0,dis_x=2, dis_y=2,dis_z=2, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   layer_num[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=0, shift_y=1, shift_z=1,dis_x=2, dis_y=2,dis_z=2, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)





def uni_xy_22_3D_sparse(tn_U, N_x_l, N_y_l,N_z_l, layer_num,list_tags,chi, list_scale,scale,seed_val=10,
 cycle="off",dist_type="uniform"):

   layer_num[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=0, shift_y=0, shift_z=0,dis_x=2, dis_y=2,dis_z=2, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   layer_num[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=0, shift_y=1, shift_z=0,dis_x=2, dis_y=2,dis_z=2, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)











def uni_xy_33_3D(tn_U, N_x_l, N_y_l,N_z_l, layer_num,list_tags,chi, list_scale,scale,seed_val=10,
 cycle="off",dist_type="uniform"):

   layer_num[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=2, shift_y=1, shift_z=1,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)
   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y,size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l,L_z=N_z_l, 
   Iso_on=True, shift_x=1, shift_y=2, shift_z=1,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)
   layer_num[0]+=1

   size_U_x=1
   size_U_y=1
   size_U_z=2
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y,size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l,L_z=N_z_l, 
   Iso_on=True, shift_x=1, shift_y=1, shift_z=2,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val,scale=scale,dist_type=dist_type)
   layer_num[0]+=1



def uni_xy_33_3D_center_zy(tn_U, N_x_l, N_y_l,N_z_l, layer_num,list_tags,chi, list_scale,scale,seed_val=10,
 cycle="off",dist_type="uniform"):

   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=1, shift_y=0, shift_z=1,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)
  

   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=1, shift_y=1, shift_z=1,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)



def uni_xy_33_3D_center_zx(tn_U, N_x_l, N_y_l,N_z_l, layer_num,list_tags,chi, list_scale,scale,seed_val=10,
 cycle="off",dist_type="uniform"):

   layer_num[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=1, shift_y=0, shift_z=1,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)
  

   layer_num[0]+=1
   size_U_x=1
   size_U_y=1
   size_U_z=2
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=1, shift_y=1, shift_z=1,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   layer_num[0]+=1
   size_U_x=2
   size_U_y=1
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=1, shift_y=2, shift_z=1,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)







def uni_xy_33_3D_corner_sparse(tn_U, N_x_l, N_y_l,N_z_l, layer_num,list_tags,chi, list_scale,scale,seed_val=10,
 cycle="off",dist_type="uniform"):

   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=0, shift_y=0, shift_z=0,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)
  

   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=2, shift_y=0, shift_z=0,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=0, shift_y=1, shift_z=2,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)
  

   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, shift_x=2, shift_y=1, shift_z=2,dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


def uni_xy_33_3D_corner_full(tn_U, N_x_l, N_y_l,N_z_l, layer_num,list_tags,chi, list_scale,scale,seed_val=10,
 cycle="off",dist_type="uniform"):

   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, 
   shift_x=0, shift_y=0, shift_z=0, dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)
  

   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, 
   shift_x=2, shift_y=0, shift_z=0, dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, 
   shift_x=0, shift_y=0, shift_z=2, dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, 
   shift_x=2, shift_y=0, shift_z=2, dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, 
   shift_x=0, shift_y=1, shift_z=0, dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)
  

   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, 
   shift_x=2, shift_y=1, shift_z=0, dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)


   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, 
   shift_x=0, shift_y=1, shift_z=2, dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)

   layer_num[0]+=1
   size_U_x=1
   size_U_y=2
   size_U_z=1
   Uni_Tn_3D(tn_U, chi, size_U_x, size_U_y, size_U_z,list_tags, list_scale,layer=layer_num[0], 
   L_x=N_x_l, L_y=N_y_l, L_z=N_z_l, 
   Iso_on=True, 
   shift_x=2, shift_y=1, shift_z=2, dis_x=3, dis_y=3,dis_z=3, 
   cycle=cycle,seed_val=seed_val, scale=scale,dist_type=dist_type)










def  Iso_33_3D(tn_U, N_x_l, N_y_l,N_z_l,layer_num,chi,seed_val=10,last_bond="off"):
  size_Iso_x=0
  size_Iso_y=3
  size_Iso_z=0

  l_x=N_x_l
  l_y=N_y_l
  l_z=N_z_l


  iso_Tn_3D(tn_U,chi, size_Iso_x, size_Iso_y,size_Iso_z,layer=3*layer_num, L_x=l_x,L_y=l_y,L_z=l_z, Iso_on=True,
         dis_x=1, dis_y=3, dis_z=1, cycle="on",seed_val=seed_val)


  size_Iso_x=3
  size_Iso_y=0
  size_Iso_z=0

  l_x=N_x_l
  l_y=N_y_l//3
  l_z=N_z_l
  
  iso_Tn_3D(tn_U,chi, size_Iso_x, size_Iso_y, size_Iso_z,layer=3*layer_num+1, L_x=l_x,L_y=l_y,L_z=l_z, Iso_on=True,dis_x=3, dis_y=1,dis_z=1, cycle="on",seed_val=seed_val)



  size_Iso_x=0
  size_Iso_y=0
  size_Iso_z=3
  l_x=N_x_l//3
  l_y=N_y_l//3
  l_z=N_z_l
  iso_Tn_3D(tn_U,chi, size_Iso_x, size_Iso_y, size_Iso_z,layer=3*layer_num+2, L_x=l_x,L_y=l_y,L_z=l_z, Iso_on=True,dis_x=1, dis_y=1,dis_z=3, cycle="on",seed_val=seed_val, last_bond=last_bond)
  layer_num[2]+=1



def  Iso_22_3D(tn_U, N_x_l, N_y_l,N_z_l,layer_num,list_tags,chi,list_scale,scale,seed_val=10,last_bond="off",
Iso=True,dist_type="uniform",data_type="float64"):


  size_Iso_x=0
  size_Iso_y=2
  size_Iso_z=0

  l_x=N_x_l
  l_y=N_y_l
  l_z=N_z_l

  layer_num[1]+=1
  iso_Tn_3D(tn_U,chi, size_Iso_x, size_Iso_y,size_Iso_z,list_tags,list_scale,layer=layer_num[1], L_x=l_x,L_y=l_y,L_z=l_z,
      dis_x=1, dis_y=2, dis_z=1, cycle="off",seed_val=seed_val,scale=scale, Iso=Iso,dist_type=dist_type,data_type=data_type)

  layer_num[1]+=1
  size_Iso_x=2
  size_Iso_y=0
  size_Iso_z=0

  l_x=N_x_l
  l_y=N_y_l//2
  l_z=N_z_l


  iso_Tn_3D(tn_U,chi, size_Iso_x, size_Iso_y, size_Iso_z,list_tags,list_scale,layer=layer_num[1], L_x=l_x,L_y=l_y,L_z=l_z,
  dis_x=2, dis_y=1,dis_z=1, cycle="off",seed_val=seed_val,scale=scale,Iso=Iso,dist_type=dist_type,data_type=data_type)
  layer_num[1]+=1


  size_Iso_x=0
  size_Iso_y=0
  size_Iso_z=2
  l_x=N_x_l//2
  l_y=N_y_l//2
  l_z=N_z_l
  iso_Tn_3D(tn_U,chi, size_Iso_x, size_Iso_y, size_Iso_z,list_tags,list_scale,layer=layer_num[1], L_x=l_x,L_y=l_y,L_z=l_z,dis_x=1, 
  dis_y=1,dis_z=2, cycle="off",seed_val=seed_val, last_bond=last_bond,scale=scale,Iso=Iso,dist_type=dist_type,data_type=data_type)
  layer_num[1]+=1







def Iso_22(tn_U, N_x_l, N_y_l,chi, num_layer,list_tags_I,label_list,shared_I,lsit_scale,seed_val=10,last_bond="off", data_type="float64",scale=0, Iso=True, dist_type="uniform"):
  size_Iso_x=0
  size_Iso_y=2
  l_x=N_x_l
  l_y=N_y_l
  num_layer[1]+=1
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,list_tags_I,shared_I,lsit_scale,layer=num_layer[1], L_x=l_x,L_y=l_y, Iso=Iso,shift_x=0, shift_y=0,
         dis_x=1, dis_y=2, cycle="on",seed_val=seed_val,data_type=data_type,scale=scale, label=label_list[0],dist_type=dist_type)

  num_layer[1]+=1
  label_list[0]+=1
  size_Iso_x=2
  size_Iso_y=0
  l_x=N_x_l
  l_y=N_y_l//2
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,list_tags_I,shared_I,lsit_scale,layer=num_layer[1], L_x=l_x,L_y=l_y, Iso=Iso,shift_x=0, shift_y=0,
          dis_x=2, dis_y=1, last_bond=last_bond, cycle="on",seed_val=seed_val,data_type=data_type,scale=scale,label=label_list[0],dist_type=dist_type)
  num_layer[1]+=1
  label_list[0]+=1



def Iso_22_y(tn_U, N_x_l, N_y_l,layer_num,chi,seed_val=10,last_bond="off"):
  size_Iso_x=0
  size_Iso_y=2
  l_x=N_x_l
  l_y=N_y_l
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,layer=layer_num, L_x=l_x,L_y=l_y, Iso_on=True,shift_x=0, shift_y=0,
         dis_x=1, dis_y=2, cycle="on",seed_val=seed_val,scale=0)


def Iso_22_x(tn_U, N_x_l, N_y_l,layer_num,chi,seed_val=10,last_bond="off"):
  size_Iso_x=2
  size_Iso_y=0
  l_x=N_x_l
  l_y=N_y_l
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,layer=layer_num, L_x=l_x,L_y=l_y, Iso_on=True,shift_x=0, shift_y=0,
          dis_x=2, dis_y=1, last_bond=last_bond, cycle="on",seed_val=seed_val)







def Iso_33_y(tn_U, N_x_l, N_y_l,layer_num,chi,seed_val=10,last_bond="off"):
  size_Iso_x=0
  size_Iso_y=3
  l_x=N_x_l
  l_y=N_y_l
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,layer=layer_num, L_x=l_x,L_y=l_y, Iso_on=True,shift_x=0, shift_y=0,
         dis_x=1, dis_y=3, cycle="on",seed_val=seed_val)


def Iso_33_x(tn_U, N_x_l, N_y_l,layer_num,chi,seed_val=10,last_bond="off"):
  size_Iso_x=3
  size_Iso_y=0
  l_x=N_x_l
  l_y=N_y_l
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,layer=layer_num, L_x=l_x,L_y=l_y, Iso_on=True,shift_x=0, shift_y=0,
          dis_x=3, dis_y=1, last_bond=last_bond, cycle="on",seed_val=seed_val)









def Iso_44(tn_U, N_x_l, N_y_l,layer_num,chi,seed_val=10, last_bond="off"):


  size_Iso_x=0
  size_Iso_y=2
  l_x=N_x_l
  l_y=N_y_l
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,layer=4*layer_num, L_x=l_x,L_y=l_y, Iso_on=True,shift_x=0, shift_y=0,
         dis_x=1, dis_y=2, cycle="on",seed_val=10)



  size_Iso_x=2
  size_Iso_y=0
  l_x=N_x_l
  l_y=N_y_l//2
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,layer=4*layer_num+1, L_x=l_x,L_y=l_y, Iso_on=True,shift_x=0, shift_y=0,
         dis_x=2, dis_y=1, last_bond="off", cycle="on",seed_val=10)


  size_Iso_x=0
  size_Iso_y=2
  l_x=N_x_l//2
  l_y=N_y_l//2
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,layer=4*layer_num+2, L_x=l_x,L_y=l_y, Iso_on=True,shift_x=0, shift_y=0,
         dis_x=1, dis_y=2, cycle="on",seed_val=10)



  size_Iso_x=2
  size_Iso_y=0
  l_x=N_x_l//2
  l_y=N_y_l//4
  iso_Tn(tn_U,chi, size_Iso_x, size_Iso_y,layer=4*layer_num+3, L_x=l_x,L_y=l_y, Iso_on=True,shift_x=0, shift_y=0,
         dis_x=2, dis_y=1, last_bond=last_bond, cycle="on",seed_val=10)






def Iso_4_3_local_3D(tn,cor,chi,num_layer,list_tags,list_scale,seed_val=10, data_type="float64",Iso=True, scale=0,dist_type="uniform"):

   list_scale.append(f"SI{scale}")
   layer=num_layer[1]
   i,j,k=cor
   for j_iter in range(4):
       num_layer[1]+=1
       layer=num_layer[1]
       where=[ (i+1,j+j_iter,k+1),(i+1,j+j_iter,k+2) ]
       dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
       dims.insert(0, min(prod(dims), chi))
       list_tags.append(f"I{layer},I{i+1},{j+j_iter},{k+1}")
       tn.reverse_gate(
          G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+j+j_iter ), 
          where=where,
          iso=Iso,
          tags=["I", f"I{layer}",f"I{layer},I{i+1},{j+j_iter},{k+1}", f"SI{scale}"],
          new_sites=[(f"{-1}",f"{-2}",f"{-3}")]
       )

       num_layer[1]+=1
       layer=num_layer[1]
       where=[ (i+2,j+j_iter,k+1),(i+2,j+j_iter,k+2) ]
       dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
       dims.insert(0, min(prod(dims), chi))
       list_tags.append(f"I{layer},I{i+2},{j+j_iter},{k+1}")
       tn.reverse_gate(
          G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+j ), 
          where=where,
          iso=Iso,
          tags=["I", f"I{layer}",f"I{layer},I{i+2},{j+j_iter},{k+1}",f"SI{scale}"],
          new_sites=[(f"{-4}",f"{-5}",f"{-6}")]
       )

       num_layer[1]+=1
       layer=num_layer[1]
       where=[ (-1,-2,-3),(-4,-5,-6) ]
       dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
       dims.insert(0, min(prod(dims), chi))
       list_tags.append(f"I{layer},I{-1},{-2},{-3}")
       tn.reverse_gate(
          G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+i+j ), 
          where=where,
          iso=Iso,
          tags=["I", f"I{layer}",f"I{layer},I{-1},{-2},{-3}",f"SI{scale}"],
          new_sites=[(f"{i+1}",f"{j+j_iter}",f"{k+1}")]
       )

       num_layer[1]+=1
       layer=num_layer[1]
       where=[ (i,j+j_iter,k+1),(i,j+j_iter,k+2) ]
       dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
       dims.insert(0, min(prod(dims), chi))
       list_tags.append(f"I{layer},I{i},{j+j_iter},{k+1}")
       tn.reverse_gate(
          G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+4*j_iter ), 
          where=where,
          iso=Iso,
          tags=["I", f"I{layer}",f"I{layer},I{i},{j+j_iter},{k+1}",f"SI{scale}"],
          new_sites=[(f"{i}",f"{j+j_iter}",f"{k+1}")]
       )

       num_layer[1]+=1
       layer=num_layer[1]
       where=[ (i+3,j+j_iter,k+1),(i+3,j+j_iter,k+2) ]
       dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
       dims.insert(0, min(prod(dims), chi))
       list_tags.append(f"I{layer},I{i+3},{j+j_iter},{k+1}")
       tn.reverse_gate(
          G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+3*j_iter+3*j ), 
          where=where,
          iso=Iso,
          tags=["I", f"I{layer}",f"I{layer},I{i+3},{j+j_iter},{k+1}",f"SI{scale}"],
          new_sites=[(f"{i+3}",f"{j+j_iter}",f"{k+1}")]
       )

       num_layer[1]+=1
       layer=num_layer[1]
       where=[ (i+1,j+j_iter,k),(i+2,j+j_iter,k) ]
       dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
       dims.insert(0, min(prod(dims), chi))
       list_tags.append(f"I{layer},I{i+1},{j+j_iter},{k}")
       tn.reverse_gate(
          G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+2*j_iter ), 
          where=where,
          iso=Iso,
          tags=["I", f"I{layer}",f"I{layer},I{i+1},{j+j_iter},{k}",f"SI{scale}"],
          new_sites=[(f"{i+1}",f"{j+j_iter}",f"{k}")]
       )

       num_layer[1]+=1
       layer=num_layer[1]
       where=[ (i+1,j+j_iter,k+3),(i+2,j+j_iter,k+3) ]
       dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
       dims.insert(0, min(prod(dims), chi))
       list_tags.append(f"I{layer},I{i+1},{j+j_iter},{k+3}")
       tn.reverse_gate(
          G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+j_iter ), 
          where=where,
          iso=Iso,
          tags=["I", f"I{layer}",f"I{layer},I{i+1},{j+j_iter},{k+3}",f"SI{scale}"],
          new_sites=[(f"{i+1}",f"{j+j_iter}",f"{k+3}")]
       )



   for i_iter in [ i+0,i+1,i+3 ]:
    for k_iter in [ k+0,k+1,k+3 ]:
     num_layer[1]+=1
     layer=num_layer[1]
     where=[ (i_iter,j+1,k_iter),(i_iter,j+2,k_iter) ]
     dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
     dims.insert(0, min(prod(dims), chi))
     list_tags.append(f"I{layer},I{i_iter},{j+1},{k_iter}")
     tn.reverse_gate(
        G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+2*i_iter+3*j_iter ), 
        where=where,
        iso=Iso,
        tags=["I", f"I{layer}",f"I{layer},I{i_iter},{j+1},{k_iter}",f"SI{scale}"],
        new_sites=[(f"{i_iter}",f"{j+1}",f"{k_iter}")]
     )


   num_layer[1]+=1
   where_list=[]
   for j_iter in [0,1,3]:
    where_list=where_list+[(0,j_iter,0),(1,j_iter,0), (3,j_iter,0), (0,j_iter,1), (1,j_iter,1), (3,j_iter,1),
   (0,j_iter,3),(1,j_iter,3),(3,j_iter,3) ]
   index_map={}
   for cor_list in where_list:
     i_l, j_l, k_l=cor_list
     i_f=i+i_l
     j_f=j+j_l
     k_f=k+k_l
     index_map[f"l{i_f},{j_f},{k_f}"] =f"l{i_f-(i_f+1)//4},{j_f-(j_f+1)//4},{k_f-(k_f+1)//4}" 

   ##print (index_map)
   tn.reindex_(index_map)







def Iso_mps(tn,chi,num_layer, list_tags,list_scale,seed_val=10, data_type="float64",Iso=True, scale=0,dist_type="uniform"):

      Lx=tn.Lx
      Ly=tn.Ly
      #print (  (0,0)  )
      list_scale.append(f"SI{scale}")
      layer=num_layer[1]
      num_layer[1]+=1
      layer=num_layer[1]
      where=[ (0,0)      ]
      dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
      dims.insert(0, min(prod(dims), chi))
      list_tags.append(f"I{layer},I{0},{0}")
      tn.reverse_gate(
         G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+0 ), 
         where=where,
         iso=Iso,
         tags=["I", f"I{layer}",f"I{layer},I{0},{0}", f"SI{scale}"],
         new_sites=[(f"{0}",f"{0}")]
      )



      for i, j in itertools.product(range(Lx), range(Ly)):
               layer=num_layer[1]
               num_layer[1]+=1
               layer=num_layer[1]
               if j != Ly-1:
                  #print (  (i,j), (i,j+1)  )
                  where=[ (i,j),(i,j+1) ]
                  dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
                  dims.insert(0, min(prod(dims), chi))
                  list_tags.append(f"I{layer},I{i},{j+1}")
                  tn.reverse_gate(
                     G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+i ), 
                     where=where,
                     iso=Iso,
                     tags=["I", f"I{layer}",f"I{layer},I{i},{j+1}", f"SI{scale}"],
                     new_sites=[(f"{i}",f"{j+1}")]
                  )
               elif i !=Lx-1:
                  #print (  (i,j),(i+1,(j+1)%Ly)  )
                  where=[ (i,j),(i+1,(j+1)%Ly) ]
                  dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
                  dims.insert(0, min(prod(dims), chi))
                  list_tags.append(f"I{layer},I{i+1},{(j+1)%Ly}")
                  tn.reverse_gate(
                     G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+i ), 
                     where=where,
                     iso=Iso,
                     tags=["I", f"I{layer}",f"I{layer},I{i+1},{(j+1)%Ly}", f"SI{scale}"],
                     new_sites=[(f"{i+1}",f"{(j+1)%Ly}")]
                  )
   
                          
      layer=num_layer[1]
      num_layer[1]+=1
      layer=num_layer[1]
      where=[ (Lx-1,Ly-1)      ]
      dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
      list_tags.append(f"I{layer},I{Lx-1},{Ly-1}")
      tn.reverse_gate(
         G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+0 ), 
         where=where,
         iso=Iso,
         tags=["I", f"I{layer}",f"I{layer},I{Lx-1},{Ly-1}", f"SI{scale}"]
      )








def Iso_4_3_local(tn,cor,chi,num_layer, list_tags,label_list,shared_I,list_scale,seed_val=10, data_type="float64",Iso=True, scale=0,dist_type="uniform"):


        list_scale.append(f"SI{scale}")
        label=label_list[0]
        where_list=[]
        layer=num_layer[1]
        i,j=cor
        where=[ (i+1,j),(i+2,j)      ]
        where_list.append( (1,0) )
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i+1},{j}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+i ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i+1},{j}", f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{i+1}",f"{j}")]
        )
        
        
        num_layer[1]+=1
        layer=num_layer[1]
        label+=1
        label_list[0]+=1
        i,j=cor
        where=[ (i+1,j+3),(i+2,j+3)      ]
        where_list.append( (1,3) )
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i+1},{j+3}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+j ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i+1},{j+3}",f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{i+1}",f"{j+3}")]
        )


        num_layer[1]+=1
        layer=num_layer[1]
        label+=1
        label_list[0]+=1

        i,j=cor
        where=[ (i,j+1),(i,j+2)      ]
        where_list.append( (0,1) )
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i},{j+1}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+j+i+20 ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i},{j+1}",f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{i}",f"{j+1}")]
        )


        num_layer[1]+=1
        layer=num_layer[1]
        label+=1
        label_list[0]+=1

        i,j=cor
        where=[ (i+3,j+1),(i+3,j+2)      ]
        where_list.append( (3,1) )
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i+3},{j+1}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+j+3*i+60 ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i+3},{j+1}",f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{i+3}",f"{j+1}")]
        )


        num_layer[1]+=1
        layer=num_layer[1]
        label+=1
        label_list[0]+=1

        i,j=cor
        where=[ (i+1,j+1),(i+1,j+2)      ]
        where_list.append( (1,1) )
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i+1},{j+1}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+4*j+3*i+90 ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i+1},{j+1}",f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{-100}",f"{-200}")]
        )



        num_layer[1]+=1
        layer=num_layer[1]
        label+=1
        label_list[0]+=1

        i,j=cor
        where=[ (i+2,j+1),(i+2,j+2)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i+2},{j+1}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+4*j+6*i+120 ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i+2},{j+1}",f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{-300}",f"{-400}")]
        )



        num_layer[1]+=1
        layer=num_layer[1]
        label+=1
        label_list[0]+=1

        i,j=cor
        where=[ (-100,-200),(-300,-400)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{-100},{-200}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+10*j+6*i+10 ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{-100},{-200}",f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{i+1}",f"{j+1}")]
        )
        where_list=where_list+[ (0,0),(0,3), (3,0),(3,3) ]
        index_map={}
        for cor_list in where_list:
          i_l, j_l=cor_list
          i_f=i+i_l
          j_f=j+j_l
          index_map[f"l{i_f},{j_f}"] =f"l{i_f-(i_f+1)//4},{j_f-(j_f+1)//4}" 

        label_list[0]+=1
        num_layer[1]+=1    
        tn.reindex_(index_map)















def Iso_3_2_local(tn,cor,chi,num_layer,list_tags,label_list,shared_I,list_scale,seed_val=10, data_type="float64", scale=0, Iso=True,dist_type="uniform"):
        ##print (data_type)
        list_scale.append(f"SI{scale}")
        i,j=cor
        label=label_list[0]
        layer=num_layer[1]
        where=[ (i,j+1),(i,j+2)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i},{j+1}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+i+j ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i},{j+1}", f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{i}",f"{j+1}")]
        )
        
        
        num_layer[1]+=1
        layer=num_layer[1]
        label+=1
        label_list[0]+=1
        i,j=cor
        where=[ (i+1,j),(i+1,j+1)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i+1},{j}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+i+1 ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i+1},{j}",f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{i+1}",f"{j}")]
        )



        num_layer[1]+=1
        layer=num_layer[1]
        label+=1
        label_list[0]+=1

        i,j=cor
        where=[ (i+2,j+1),(i+2,j+2)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i+2},{j+1}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+7+j ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i+2},{j+1}",f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{i+2}",f"{j+1}")]
        )


        num_layer[1]+=1
        layer=num_layer[1]
        label+=1
        label_list[0]+=1


        i,j=cor
        where=[ (i+1,j+2),(i+2,j+1)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i+1},{j+2}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+9+i), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i+1},{j+2}",f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{i+1}",f"{j+2}")]
        )


        num_layer[1]+=1
        layer=num_layer[1]
        label+=1
        label_list[0]+=1

        i,j=cor
        where=[ (i,j),(i+1,j)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i},{j}")
        shared_I.append(f"SI{scale},I{label}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+i+j ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i},{j}",f"SI{scale}",f"SI{scale},I{label}"],
           new_sites=[(f"{i}",f"{j}")]
        )

        num_layer[1]+=1
        label_list[0]+=1
        where_list=[(0,0),(2,0),(0,1),(1,2) ]
        index_map={}
        for cor_list in where_list:
          i_l, j_l=cor_list
          i_f=i+i_l
          j_f=j+j_l
          index_map[f"l{i_f},{j_f}"] =f"l{i_f-(i_f+1)//3},{j_f-(j_f+1)//3}" 

    
        ##print (index_map)
        tn.reindex_(index_map)



def Iso_3_2_local_a(tn,cor, layer,chi,seed_val=10, data_type="float64",Iso_on=True):
        
        i,j=cor
        where=[ (i,j),(i,j+1)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist="uniform", seed=seed_val ), 
           where=where,
           iso=Iso_on,
           tags=["I", f"I{layer}",f"Ilay{layer},k{i},{j}"],
           new_sites=[(f"{i}",f"{j}")]
        )
        
        
        
        i,j=cor
        where=[ (i+1,j),(i+1,j+1)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist="uniform", seed=seed_val ), 
           where=where,
           iso=Iso_on,
           tags=["I", f"I{layer}",f"Ilay{layer},k{i},{j+3}"],
           new_sites=[(f"{i+1}",f"{j}")]
        )




        i,j=cor
        where=[ (i+2,j),(i+2,j+1)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist="uniform", seed=seed_val ), 
           where=where,
           iso=Iso_on,
           tags=["I", f"I{layer}",f"Ilay{layer},k{i},{j+1}"],
           new_sites=[(f"{i+2}",f"{j}")]
        )




        i,j=cor
        where=[ (i+1,j),(i+2,j)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist="uniform", seed=seed_val ), 
           where=where,
           iso=Iso_on,
           tags=["I", f"I{layer}",f"Ilay{layer},k{i+3},{j+1}"],
           new_sites=[(f"{i+1}",f"{j}")]
        )



        i,j=cor
        where=[ (i+1,j+2),(i+2,j+2)      ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist="uniform", seed=seed_val ), 
           where=where,
           iso=Iso_on,
           tags=["I", f"I{layer}",f"Ilay{layer},k{i+1},{j+1}"],
           new_sites=[(f"{i+1}",f"{j+2}")]
        )



        index_map={}
        for i_l in range(3):
         for j_l in range(3):
          i_f=i+i_l
          j_f=j+j_l
          index_map[f"l{i_f},{j_f}"] =f"l{i_f-(i_f+1)//3},{j_f-(j_f+1)//3}" 

    
        ##print (index_map)
        tn.reindex_(index_map)

def  Iso_33_11(tn_U, N_x, N_y,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,scale=0,uni_h="off",uni_h_full="off",uni_d="off",last_bond="off",cycle="off",data_type="float64",
Iso=True,Iso_apply="on",dist_type="uniform"):

 label_list=[0]
 label_listU=[0]


 if uni_h=="on":
  uni_xy_33(tn_U, N_x, N_y, chi, num_layer, list_tags_U,label_listU,shared_U,list_scale, scale=scale,seed_val=0, cycle=cycle,data_type=data_type,dist_type=dist_type)


 if uni_h_full=="on":
  uni_xy_33_full(tn_U, N_x, N_y, chi, num_layer, list_tags_U,label_listU,shared_U,list_scale, scale=scale,seed_val=0, cycle=cycle,data_type=data_type,dist_type=dist_type)


 if uni_d=="on":
  uni_dense_22(tn_U, N_x, N_y, chi, num_layer, list_tags_U,label_listU,shared_U, list_scale, scale=scale,seed_val=0, cycle=cycle,data_type=data_type,dist_type=dist_type)


 if Iso_apply=="on":
  for i in range(0,N_x,3):
   for j in range(0,N_y,3):
    cor=(i,j)
    layer=1
    if i==0 and j==0:
     Iso_3_2_local(tn_U,cor, chi, num_layer, list_tags_I,label_list,shared_I,list_scale,scale=scale,seed_val=20,data_type=data_type,Iso=Iso,dist_type=dist_type)
     #Iso_3_2_local_a(tn_U,cor, layer,chi,seed_val=10)
    else:
     empty=[0] 
     Iso_3_2_local(tn_U,cor, chi, num_layer, list_tags_I,empty,empty,empty,scale=scale,seed_val=10,data_type=data_type, Iso=Iso,dist_type=dist_type)


  N_x=N_x-N_x//3
  N_y=N_y-N_y//3
  Iso_22(tn_U, N_x, N_y,chi,num_layer, list_tags_I,label_list,shared_I,list_scale,scale=scale,seed_val=20, last_bond=last_bond,data_type=data_type, Iso=Iso,dist_type=dist_type)





def  Iso_44_11_3D(tn_U, N_x, N_y, N_z,chi,num_layer,list_tags_U,list_tags_I, shared_I, shared_U, list_scale,
scale=0, uni_h="off", uni_top="off", uni_h_Iso="off",uni_h_Iso_top="off",
last_bond="off", cycle="off", data_type="float64", Iso_apply="on", seed_val=10,
dist_type="uniform", seed_val_u=0, Iso=True, uni_h_Iso_top_top="on"):


 if uni_h=="on":
    uni_xy_44_3D(tn_U, N_x, N_y,N_z,chi,num_layer,list_tags_U, list_scale, scale, seed_val=seed_val_u, cycle=cycle, 
                   dist_type=dist_type)
    #uni_xy_44_3D_sparse(tn_U, N_x, N_y,N_z,chi,num_layer,list_tags_U, list_scale, scale, seed_val=seed_val_u, cycle=cycle, 
     #  dist_type=dist_type)
    #uni_xy_44_3D_full(tn_U, N_x, N_y,N_z,chi,num_layer,list_tags_U, list_scale, scale, seed_val=seed_val_u, cycle=cycle, 
     # dist_type=dist_type)


 if uni_h_Iso=="on":
   uni_xy_44_3D_iso_corner(tn_U, N_x, N_y,N_z,chi,num_layer,list_tags_U, list_scale, scale, seed_val=seed_val_u, cycle=cycle,
                  dist_type=dist_type)
   #uni_xy_44_3D_iso_corner_full(tn_U, N_x, N_y,N_z,chi,num_layer,list_tags_U, list_scale, scale, 
    #               seed_val=seed_val_u, cycle="off",
    #                  dist_type=dist_type)
   #uni_xy_44_3D_iso_center_zx(tn_U, N_x, N_y,N_z,chi,num_layer,list_tags_U, list_scale, scale, seed_val=seed_val_u, cycle=cycle, 
    #               dist_type=dist_type)
   #uni_xy_44_3D_iso_center_zy(tn_U, N_x, N_y,N_z,chi,num_layer,list_tags_U, list_scale, scale, seed_val=seed_val_u, cycle=cycle,
    #                dist_type=dist_type)


 if Iso_apply=="on":
  for i in range(0,N_x,4):
   for j in range(0,N_y,4):
    for k in range(0,N_z,4):
       cor=(i,j,k)
       Iso_4_3_local_3D(tn_U,cor,chi,num_layer,list_tags_I,list_scale,scale=scale,seed_val=seed_val,
                     data_type=data_type,Iso=Iso,dist_type=dist_type)

 N_x=N_x-N_x//4 
 N_y=N_y-N_y//4
 N_z=N_z-N_z//4

 if uni_top=="on":
      uni_xy_33_3D(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
                    cycle=cycle,dist_type=dist_type)



 if uni_h_Iso_top=="on":
      #uni_xy_33_3D_corner_full(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
       #             cycle="off",dist_type=dist_type)
      uni_xy_33_3D_center_zy(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
                     cycle="off",dist_type=dist_type)
      #uni_xy_33_3D_center_zx(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
       #              cycle="off",dist_type=dist_type)


 if Iso_apply=="on":

  for i in range(0,N_x,3):
   for j in range(0,N_y,3):
    for k in range(0,N_z,3):
      cor=(i,j,k)
      Iso_3_2_local_3D(tn_U,cor,num_layer,list_tags_I,chi,list_scale,scale,
                      seed_val=seed_val,data_type=data_type,Iso=Iso,dist_type=dist_type)

 
 N_x=N_x-N_x//3
 N_y=N_y-N_y//3
 N_z=N_z-N_z//3


 if uni_h_Iso_top_top=="on":
      #uni_xy_22_3D_full(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
       #             cycle="off",dist_type=dist_type)
      uni_xy_22_3D_sparse(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
                     cycle="off",dist_type=dist_type)

 
 if Iso_apply=="on":
      Iso_22_3D(tn_U, N_x, N_y,N_z,num_layer,list_tags_I,chi,list_scale,scale,seed_val=seed_val, 
                    last_bond=last_bond,dist_type=dist_type,data_type=data_type)










def  Iso_33_11_3D(tn_U, N_x, N_y, N_z,chi,num_layer,list_tags_U,list_tags_I, shared_I, shared_U, list_scale,
scale=0, uni_h="off", uni_h_Iso="off",uni_h_Iso_top="off",
last_bond="off", cycle="off", data_type="float64", Iso_apply="on", seed_val=10,
dist_type="uniform", seed_val_u=0, Iso=True, uni_h_Iso_top_top="on"):


 if uni_h=="on":
      uni_xy_33_3D(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
                    cycle=cycle,dist_type=dist_type)



 if uni_h_Iso=="on":
      #uni_xy_33_3D_corner_full(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
       #             cycle=cycle,dist_type=dist_type)
      uni_xy_33_3D_center_zy(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
                     cycle=cycle,dist_type=dist_type)
      #uni_xy_33_3D_center_zx(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
      #               cycle=cycle,dist_type=dist_type)


 if Iso_apply=="on":

  for i in range(0,N_x,3):
   for j in range(0,N_y,3):
    for k in range(0,N_z,3):
      cor=(i,j,k)
      Iso_3_2_local_3D(tn_U,cor,num_layer,list_tags_I,chi,list_scale,scale,
                      seed_val=seed_val,data_type=data_type,Iso=Iso,dist_type=dist_type)

 
 N_x=N_x-N_x//3
 N_y=N_y-N_y//3
 N_z=N_z-N_z//3


 if uni_h_Iso_top=="on":
      uni_xy_22_3D_full(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
                    cycle=cycle,dist_type=dist_type)
      #uni_xy_22_3D_sparse(tn_U, N_x, N_y,N_z,num_layer,list_tags_U,chi,list_scale,scale, seed_val=seed_val_u, 
       #              cycle=cycle,dist_type=dist_type)

 
 if Iso_apply=="on":
      Iso_22_3D(tn_U, N_x, N_y,N_z,num_layer,list_tags_I,chi,list_scale,scale,seed_val=seed_val, 
                    last_bond=last_bond,dist_type=dist_type,data_type=data_type)







def  mps_build(tn_U, N_x, N_y,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,
list_scale,scale=0,uni_h="off",uni_h_full="off",uni_h_Iso="off",cycle="off", data_type="float64",seed_val=10,dist_type="uniform", seed_val_u=0):

 label_list=[0]
 label_listU=[0]

 if uni_h=="on":
     uni_xy_44_mps( tn_U, N_x, N_y, chi, num_layer, list_tags_U,label_listU,shared_U,list_scale, scale=scale, seed_val=seed_val_u, cycle=cycle, data_type=data_type, dist_type=dist_type)

 if uni_h_full=="on":
     uni_xy_44_full( tn_U, N_x, N_y, chi, num_layer, list_tags_U,label_listU,shared_U,list_scale, scale=scale,seed_val=seed_val_u, cycle=cycle, data_type=data_type,dist_type=dist_type)

 if uni_h_Iso=="on":
    uni_xy_44_Iso( tn_U, N_x, N_y, chi, num_layer, list_tags_U,label_listU,shared_U,list_scale, scale=scale,seed_val=seed_val_u, cycle="off", data_type=data_type,dist_type=dist_type)


 #Iso_mps(tn_U, chi, num_layer, list_tags_I,list_scale,scale=scale,seed_val=seed_val,data_type=data_type,Iso=True,dist_type=dist_type)
   
   
   
 














def  Iso_44_11(tn_U, N_x, N_y,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,
list_scale,scale=0,uni_h="off",uni_top="off",uni_h_full="off",uni_h_Iso="off",
last_bond="off",cycle="off", data_type="float64", Iso_apply="on", Iso_1=True, Iso_2=True,seed_val=10,dist_type="uniform", seed_val_u=0):

 label_list=[0]
 label_listU=[0]

 if uni_h=="on":
      uni_xy_44( tn_U, N_x, N_y, chi, num_layer, list_tags_U,label_listU,shared_U,list_scale, scale=scale, seed_val=seed_val_u, cycle=cycle, data_type=data_type, dist_type=dist_type)



 if uni_h_full=="on":
      uni_xy_44_full( tn_U, N_x, N_y, chi, num_layer, list_tags_U,label_listU,shared_U,list_scale, scale=scale,seed_val=seed_val_u, cycle=cycle, data_type=data_type,dist_type=dist_type)



 if uni_h_Iso=="on":
      uni_xy_44_Iso( tn_U, N_x, N_y, chi, num_layer, list_tags_U,label_listU,shared_U,list_scale, scale=scale,seed_val=seed_val_u, cycle="off", data_type=data_type,dist_type=dist_type)


 if Iso_apply=="on":
  for i in range(0,N_x,4):
   for j in range(0,N_y,4):
    cor=(i,j)
    if i==0 and j==0:
     Iso_4_3_local(tn_U,cor, chi, num_layer, list_tags_I,label_list, shared_I,list_scale,scale=scale,seed_val=seed_val,data_type=data_type,Iso=Iso_1,dist_type=dist_type)
    else:
     empty=[0] 
     Iso_4_3_local(tn_U,cor, chi, num_layer, list_tags_I,empty, empty,empty,scale=scale,seed_val=seed_val,data_type=data_type,Iso=Iso_1,dist_type=dist_type)
   
   
   
 N_x=N_x-N_x//4 
 N_y=N_y-N_y//4

 if uni_top=="on":
    uni_xy_33(tn_U, N_x, N_y, chi, num_layer, list_tags_U,label_listU,shared_U,list_scale, scale=scale,seed_val=seed_val_u, cycle=cycle,data_type=data_type,dist_type=dist_type)

 if Iso_apply=="on":
  for i in range(0,N_x,3):
   for j in range(0,N_y,3):
    cor=(i,j)
    layer=1
    if i==0 and j==0:
     Iso_3_2_local(tn_U,cor, chi, num_layer, list_tags_I,label_list,shared_I,list_scale,scale=scale,seed_val=seed_val,data_type=data_type,Iso=Iso_2,dist_type=dist_type)
     #Iso_3_2_local_a(tn_U,cor, layer,chi,seed_val=10)
    else:
     empty=[0] 
     Iso_3_2_local(tn_U,cor, chi, num_layer, list_tags_I,empty,empty,empty,scale=scale,seed_val=seed_val,data_type=data_type,Iso=Iso_2,dist_type=dist_type)



 N_x=N_x-N_x//3
 N_y=N_y-N_y//3

 #if uni_h_Iso=="on":
   #uni_xy_22_Iso( tn_U, N_x, N_y, chi, num_layer, list_tags_U,label_listU,shared_U,list_scale, scale=scale,seed_val=seed_val_u, cycle="off", data_type=data_type,dist_type=dist_type)


 if Iso_apply=="on":
   Iso_22(tn_U, N_x, N_y,chi,num_layer, list_tags_I,label_list,shared_I,list_scale,scale=scale,seed_val=seed_val, last_bond=last_bond,data_type=data_type,Iso=Iso_2,dist_type=dist_type)


















def Iso_3_2_local_3D(tn,cor, layer_number,list_tags,chi,list_scale,scale,dist_type="uniform",seed_val=10, data_type="float64",Iso=True):

    i,j,k=cor
    list_scale.append(f"SI{scale}")
    layer=layer_number[1]
    for i_iter in range(i,i+3,1):
     for j_iter in range(j,j+3,1):
      for k_iter in range(k,k+1,1):
        where=[ (i_iter,j_iter,k_iter),(i_iter,j_iter,k_iter+1) ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i_iter},{j_iter},{k_iter}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val+j_iter+k_iter ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i_iter},{j_iter},{k_iter}", f"SI{scale}"],
           new_sites=[(f"{i_iter}",f"{j_iter}",f"{k_iter}")]
        )



    layer_number[1]+=1
    layer=layer_number[1]
    for i_iter in range(i,i+3,1):
      for k_iter in [k,k+2]:
        where=[ (i_iter,j+0,k_iter),(i_iter,j+1,k_iter) ]
        dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
        dims.insert(0, min(prod(dims), chi))
        list_tags.append(f"I{layer},I{i_iter},{j},{k_iter}")
        tn.reverse_gate(
           G=qu.randn(dims, dtype=data_type, dist="uniform", seed=seed_val+k_iter+i_iter ), 
           where=where,
           iso=Iso,
           tags=["I", f"I{layer}",f"I{layer},I{i_iter},{j},{k_iter}",f"SI{scale}"],
           new_sites=[(f"{i_iter}",f"{j}",f"{k_iter}")]
        )

    layer_number[1]+=1
    layer=layer_number[1]
    for k_iter in [k,k+2]:
     for j_iter in [j,j+2]:
      where=[ (i,j_iter,k_iter),(i+1,j_iter,k_iter) ]
      dims = [tn.ind_size(tn.layer_ind(*coo)) for coo in where]
      dims.insert(0, min(prod(dims), chi))
      list_tags.append(f"I{layer},I{i},{j_iter},{k_iter}")
      tn.reverse_gate(
         G=qu.randn(dims, dtype=data_type, dist=dist_type, seed=seed_val ), 
         where=where,
         iso=Iso,
         tags=["I", f"I{layer}",f"I{layer},I{i},{j_iter},{k_iter}", f"SI{scale}"],
         new_sites=[(f"{i}",f"{j_iter}",f"{k_iter}")]
      )


    layer_number[1]+=1
    where_list=[(0,0,0),(2,0,0),(0,2,0),(2,2,0) ]
    where_list=where_list+[(0,0,2),(2,0,2),(0,2,2),(2,2,2) ]
    index_map={}
    for cor_list in where_list:
      i_l, j_l, k_l=cor_list
      i_f=i+i_l
      j_f=j+j_l
      k_f=k+k_l
      index_map[f"l{i_f},{j_f},{k_f}"] =f"l{i_f-(i_f+1)//3},{j_f-(j_f+1)//3},{k_f-(k_f+1)//3}" 

    tn.reindex_(index_map)



def   Info_contract(tn_U,list_sites,data_type="float64", opt="auto-hq"):
  list_width={}
  list_flops={}
  list_peak={}
  tn_U.unitize_(method='qr', allow_no_left_inds=True)
  for i in range(len(list_sites)):
      
      where=list_sites[i]
      tags = [ tn_U.site_tag(*coo) for coo in where ]
      tn_U_ij = tn_U.select(tags, which='any')
      tn_U_ij_G=tn_U_ij.gate(qu.pauli("I",dtype=data_type) & qu.pauli("I",dtype=data_type), list_sites[i])
      tn_U_ij_ex = (tn_U_ij_G & tn_U_ij.H)
      #tn_U_ij_ex.rank_simplify_()
      width=tn_U_ij_ex.contraction_width( optimize=opt)
      flops=np.log10(tn_U_ij_ex.contraction_cost(optimize=opt))
      tree = tn_U_ij_ex.contraction_tree(opt)
      #treec = tn.contraction_tree(copt)
      #print ( "contract", i, where, width, flops, np.log2(tree.peak_size()) )
      #print ( tn_U_ij_ex.contract(all, optimize=opt) )
      list_width[i] = width
      list_flops[i] = float(flops)
      list_peak[i]= np.log2(tree.peak_size())
      #elem_w=max(list_width, key=list_width.get)
      #elem_f=max(list_flops, key=list_flops.get)
      #print ("Max_results", list_width[elem_w], list_flops[elem_f],list_sites[elem_w],list_sites[elem_f]  )


  elem_w=max(list_width, key=list_width.get)
  elem_f=max(list_flops, key=list_flops.get)
  elem_p=max(list_peak, key=list_peak.get)
  
  print ("Max_results", list_width[elem_w], list_peak[elem_p], list_flops[elem_f],list_sites[elem_w], list_sites[elem_f]  )
  return list_width[elem_w], list_flops[elem_f], list_peak[elem_p]



def   Info_contract_test(tn_U,list_sites,data_type="float64", opt="auto-hq"):
  list_width={}
  list_flops={}
  list_peak={}
  tn_U.unitize_(method='qr', allow_no_left_inds=True)
  for i in range(len(list_sites)):
      
      where=list_sites[i]
      tags = [ tn_U.site_tag(*coo) for coo in where ]
      tn_U_ij = tn_U.select(tags, which='any')
      tn_U_ij_G=tn_U_ij.gate(qu.pauli("I",dtype=data_type) & qu.pauli("I",dtype=data_type), list_sites[i])
      tn_U_ij_ex = (tn_U_ij_G & tn_U_ij.H)
      tn_U_ij_ex.rank_simplify_()
      width=tn_U_ij_ex.contraction_width( optimize=opt)
      flops=np.log10(tn_U_ij_ex.contraction_cost(optimize=opt))
      tree = tn_U_ij_ex.contraction_tree(opt)
      #treec = tn.contraction_tree(copt)
      #print ( "contract", i, where, width, flops, np.log2(tree.peak_size()) )
      #print ( tn_U_ij_ex.contract(all, optimize=opt) )
      list_width[i] = width
      list_flops[i] = float(flops)
      list_peak[i]= np.log2(tree.peak_size())
      #elem_w=max(list_width, key=list_width.get)
      #elem_f=max(list_flops, key=list_flops.get)
      #print ("Max_results", list_width[elem_w], list_flops[elem_f],list_sites[elem_w],list_sites[elem_f]  )


  elem_w=max(list_width, key=list_width.get)
  elem_f=max(list_flops, key=list_flops.get)
  elem_p=max(list_peak, key=list_peak.get)
  
  print ("Max_results", list_width[elem_w], list_peak[elem_p], list_flops[elem_f],list_sites[elem_w], list_sites[elem_f]  )
  return list_width[elem_w], list_flops[elem_f], list_peak[elem_p]


def   Info_contract_Tree(tn_U,list_sites,data_type="float64", opt="auto-hq"):

  tn_U.unitize_(method='mgs', allow_no_left_inds=True)
  for i in range(len(list_sites)):
      
      where=list_sites[i]
      tags = [ tn_U.site_tag(*coo) for coo in where ] + ["TREE"]
      tn_U_ij = tn_U.select(tags, which='any')
      #mera_ij.draw( color=[f'lay{i}' for i in range(int(math.log2(L_L)))], iterations=600, figsize=(100, 100),node_size=700 , edge_scale=6,  initial_layout='spectral', edge_alpha=0.63333)
      #plt.savefig(f'qmera-brickwall{i}.pdf')
      #plt.clf()
      tn_U_ij_G=tn_U_ij.gate(qu.pauli("I",dtype=data_type) & qu.pauli("I",dtype=data_type), list_sites[i])
      tn_U_ij_ex = (tn_U_ij_G & tn_U_ij.H)
      tn_U_ij_ex.rank_simplify_()
      print ( "contract", i,where, tn_U_ij_ex.contraction_width( optimize=opt), tn_U_ij_ex.contraction_cost(optimize=opt) )
      print ( tn_U_ij_ex.contract(all, optimize=opt) )



def  eliminate_dupl(list_scale):

 list_scale_f = []
 for i in list_scale:
   if i not in list_scale_f:
       list_scale_f.append(i)

 return list_scale_f

def Vertical_mat(dim_lind, chi_n,chi_o,rand_strength):
  array_1=np.zeros(dim_lind*(chi_n-chi_o)).reshape(dim_lind, chi_n-chi_o)
  ran_np=qu.rand(dim_lind*(chi_n-chi_o), dist='uniform',seed=0).reshape(dim_lind, chi_n-chi_o)
  for i in range(dim_lind):
    for j in range(chi_n-chi_o):
         if i==dim_lind-1:
             array_1[i-j][j]=1.0
             
  array_1=array_1+rand_strength*ran_np
  print ("Hi", array_1.shape)
  return   array_1

def Uni_mat(new_bond_dim, chi_o,rand_strength):

  array_1=np.zeros((new_bond_dim*new_bond_dim-chi_o*chi_o)*new_bond_dim*new_bond_dim).reshape(new_bond_dim*new_bond_dim, (new_bond_dim*new_bond_dim-chi_o*chi_o))
  ran_np=qu.rand((new_bond_dim*new_bond_dim-chi_o*chi_o)*new_bond_dim*new_bond_dim, dist='uniform',seed=0).reshape(new_bond_dim*new_bond_dim, (new_bond_dim*new_bond_dim-chi_o*chi_o))

  for i in range(new_bond_dim*new_bond_dim):
   for j in range((new_bond_dim*new_bond_dim-chi_o*chi_o)):
       if i==new_bond_dim*new_bond_dim-1:
         array_1[i-j][j]=0.0     

  array_1=array_1+rand_strength*ran_np
  #print (array_1)
  return   array_1


def  expand_bond_MERA( tn_U, list_tags_I,list_tags_U, method='pad', new_bond_dim=4, rand_strength=0.01, rand_strength_u=0.01, data_type="float64"):
   for tag_id in list_tags_I:
        inds_to_expand=update_top_inds(tn_U, tag_id,new_bond_dim,rand_strength, method=method,data_type=data_type)
        while inds_to_expand:
              inds_to_expand,tag_id=update_neighbour_topinds(tn_U, tag_id,inds_to_expand,new_bond_dim,rand_strength,rand_strength_u,list_tags_U,method=method,data_type=data_type)





def  update_top_inds( tn_U, tag_id, new_bond_dim, rand_strength, method="pad",data_type="float64"): 
     tensor=tn_U[tag_id]
     l_lind=list(tensor.left_inds)
     dim_lind=np.prod([ tn_U.ind_size(i)  for i in l_lind])
     new_bond_dim_local=0

     if new_bond_dim <= dim_lind:
            new_bond_dim_local=new_bond_dim
     else:
            new_bond_dim_local=dim_lind
        
     inds_to_expand = list( set(tensor.inds) - set(tensor.left_inds) )
     chi_o=np.prod([ tn_U.ind_size(i)  for i in inds_to_expand])
     chi_n=new_bond_dim_local
     
     shape_o=[ tn_U.ind_size(i)  for i in l_lind]+[chi_n]
     
     pads = [(0, 0) if i not in inds_to_expand else
                    (0, max(new_bond_dim_local - d, 0))
                    for d, i in zip(tensor.shape, tensor.inds)]

     if rand_strength > 0 and inds_to_expand:
        edata = do('pad', tensor.data, pads, mode=rand_padder,
                   rand_strength=rand_strength)
     else:
        edata = do('pad',tensor.data,pads,mode='constant',constant_values=0)


     tensor.modify(data=edata)
     tensor.modify(left_inds=l_lind)

     return inds_to_expand


def  update_neighbour_topinds(tn_U, tag_id,common_inds, new_bond_dim,rand_strength,rand_strength_u,list_tags_U, method="pad", data_type="float64"):

     #print (tag_id)
     new_bond_dim_local=tn_U.ind_size(common_inds[0]) 
     cor_origin=tn_U.tag_map[tag_id]
     neighbour=tn_U.ind_map[common_inds[0]]-cor_origin 
     tensor=tn_U.tensor_map[list(neighbour)[0]]
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


              if method=="pad" :
                    if rand_strength_u > 0 :
                      edata = do('pad', tensor.data, pads, mode=rand_padder,
                                 rand_strength=rand_strength_u)
                    else:
                      edata = do('pad', tensor.data, pads, mode='constant')

              tensor.modify(data=edata)
              tensor.modify(left_inds=left_inds)
              tag_id_new=[ i for i in list(tensor.tags) if i in  list_tags_U]
              return send_out_ind, tag_id_new[0] #neighbour

     else:
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
             if method=="manual":
                    edata = do('pad', tensor.data, pads, mode='constant', constant_values=rand_strength)


             tensor.modify(data=edata)
             tensor.modify(left_inds=left_inds)
             ##print ("result", pads,tn_U.tensor_map[neighbour], "\n")
             return [], neighbour








def get_3d_pos(Lx, Ly, Lz, a=22, b=44, p=0.2):
   import math
   import itertools
   return {
       (i, j, k): (
           + i * math.cos(math.pi * a / 180) + j * math.cos(math.pi * b / 180) / 2**p,
           - i * math.sin(math.pi * a / 180) + j * math.sin(math.pi * b / 180) / 2**p + k       
       )
       for i, j, k in
       itertools.product(range(Lx), range(Ly), range(Lz))
   }


def  Plot_TN(tn_U, list_scale,list_tags_I, list_tags_U,phys_dim):
  fix = {
      f'reg{i},{j}': (i, j) for i, j in itertools.product(range(tn_U.Lx), range(tn_U.Ly))
  }
  fix1 = {
      f'k{i},{j}': (i+0.25, j+0.25) for i, j in itertools.product(range(tn_U.Lx), range(tn_U.Ly))
  }

  fix3 = {
      f'l{i},{j}': (i,j) for i, j in itertools.product(range(tn_U.Lx), range(tn_U.Ly))
  }


#  fix1 = {
#      f'k{i},{j}': (i+0.25, j+0.25) for i in           )
#  }


#  fix1 = {
#      f'k{i},{j}': (i+0.25, j+0.25) for i, j in itertools.product(range(tn_U.Lx), range(tn_U.Ly))
#  }

  fix.update(fix)
  fix.update(fix3)
  #custom_colors=["#FF00FF"]
  #spectral
  #list_scale=["const"]+list_scale
  marker_list=["s","P", "s", "P", "s", "P", "s", "P","s", "P"]
  hatch_list=['*', '////','*', '////', '*', '////','*', '////']
  marker_list=marker_list[:len(list_scale)]
  hatch_list=hatch_list[:len(list_scale)]

  dict_nodes = dict(zip(list_scale, marker_list))
  dict_hatch = dict(zip(list_scale, hatch_list))

  list_inds=()
  for tag in list_tags_U:
    t = tn_U[tag]
    inds = t.inds
    list_inds+=inds

  list_inds=()
  for t in tn_U["const"] :
    inds = t.left_inds
    list_inds+=inds
  
  
  print (dict_nodes,list_tags_U)
  
  
  tn_U.draw( color=list_scale,fix=fix, iterations=600, figsize=(60, 60),  
  return_fig=True,node_size=4220 , edge_scale=3, initial_layout='spectral', edge_alpha=0.53, 
  legend=True,show_tags=False, arrow_length=0.2, 
  node_shape=dict_nodes, highlight_inds=list_inds,highlight_inds_color="#ffffff",
  node_hatch=dict_hatch,node_outline_size=1.2,
  node_outline_darkness=0.2,arrow_closeness=1.0,
  #show_inds='bond-size',
  font_size=12, font_size_inner=22)
  plt.savefig('mera.pdf')
  plt.clf()




def  Plot_TN_3d(tn_U,list_scale,list_tags_I, list_tags_U, phys_dim):




  tn_3d=TN3D_rand(tn_U.Lx, tn_U.Ly, tn_U.Lz, phys_dim, cyclic=False, site_tag_id='g{},{},{}')


  fix = {
      f'g{i},{j},{k}': pos for (i, j, k) , pos in get_3d_pos( tn_U.Lx,tn_U.Ly,tn_U.Lz).items()
  }
  fix1 = {
      f'reg{i},{j},{k}': pos for (i, j, k) , pos in get_3d_pos( tn_U.Lx,tn_U.Ly,tn_U.Lz).items()
  }
  fix2 = {
      f'k{i},{j},{k}': pos for (i, j, k) , pos in get_3d_pos( tn_U.Lx,tn_U.Ly,tn_U.Lz).items()
  }

  fix3 = {
      f'l{i},{j},{k}': pos for (i, j, k) , pos in get_3d_pos( tn_U.Lx,tn_U.Ly,tn_U.Lz).items()
  }

  fix.update(fix1)
  fix.update(fix2)
  #fix.update(fix3)

  #print (tn_U)

  tn=(tn_U | tn_3d)
  
  #custom_colors=["#FF6347", "#FA8072","#CD5C5C","#E0FFFF"]
  #print (fix)
  marker_list=["s","P", "H", "X", "+", "P", "s", "P","s", "P"]
  hatch_list=['-', '////','-', '////', '-', '////','-', '////']
  marker_list=marker_list[:len(list_scale)]
  hatch_list=hatch_list[:len(list_scale)]

  dict_nodes = dict(zip(list_scale, marker_list))
  dict_hatch = dict(zip(list_scale, hatch_list))

  list_inds=()
  for tag in list_tags_U:
    t = tn_U[tag]
    inds = t.inds
    list_inds+=inds

  list_inds=()
  for t in tn_U["const"] :
    inds = t.left_inds
    list_inds+=inds
  
  
  #print (dict_nodes,list_tags_U)
  print (list_scale)
  
  tn.draw( color=list_scale,fix=fix, iterations=600, figsize=(50, 50),  
  return_fig=True,node_size=2520 , edge_scale=7, 
  initial_layout='spectral', edge_alpha=0.83, 
  legend=True,show_tags=False, arrow_length=0.5, 
  node_shape=dict_nodes, highlight_inds=list_inds,
  highlight_inds_color="#ffffff",
  node_hatch=dict_hatch,node_outline_size=1.2,
  node_outline_darkness=0.3,arrow_closeness=1.0, 
  #show_inds='bond-size',
  font_size=12, 
  font_size_inner=22)

#   tn_3d.draw( color=list_scale,fix=fix, iterations=600, figsize=(50, 50),  
#   return_fig=True,node_size=4520 , edge_scale=7, 
#   initial_layout='spectral', edge_alpha=0.83, 
#   legend=True,show_tags=False, arrow_length=0.5, highlight_inds=list_inds,
#   highlight_inds_color="#ffffff",node_outline_size=1.2,
#   node_outline_darkness=0.3,arrow_closeness=1.0, 
#   #show_inds='bond-size',
#   font_size=12, 
#   font_size_inner=22)


  plt.savefig('mera3d.pdf')
  plt.clf()

#circular
#spectral
def   Tree_to_ISOlike(tn_U, tn_tree,list_tags_I, list_tags_U):
  for i in list_tags_I:
         left_inds_t=tn_U[i].left_inds
         print (left_inds_t)
         e_data=tn_tree[i].data
         tn_U[i].modify(data=e_data)
         tn_U[i].modify(left_inds=left_inds_t)
  return tn_U


def  Init_TN(tn_U, list_tags_I, list_tags_U):


  tn=load_from_disk("Store/tn_U")
  for i in list_tags_I:

           old_tags=tn_U[i].tags
           if  'TREE' in old_tags:
                left_inds_t=tn[i].left_inds
                print (left_inds_t)
                tn[i].modify(tags=list(tn_U[i].tags)+['TREE'])
                e_data=tn[i].data
                tn[i].modify(data=e_data)
                tn[i].modify(left_inds=left_inds_t)
  return tn





def   TN_to_iso(tn, list_tags_I,list_tags_U):

  for i in list_tags_I:
        tensor=tn[i]
        left_inds = tensor.left_inds
        L_inds = list(left_inds)
        R_inds = [ix for ix in tensor.inds if ix not in L_inds]

        LR_inds = L_inds + R_inds
        tensor.transpose_(*LR_inds)
        # fuse this tensor into a matrix and 'isometrize' it
        x = tensor.to_dense(L_inds, R_inds)
        (u,s,vh)=qu.svd(x, return_vecs=True)
        #print (x.shape, u.shape, vh.shape)
        #print (x, u@ np.diag(s) @ vh)
        edata=u @ vh
        #print (np.linalg.norm(x-edata))

        edata=edata.reshape(tensor.shape)
        tensor.modify(data=edata)
        tensor.modify(left_inds=left_inds)



  for i in list_tags_U:
        tensor=tn[i]
        left_inds = tensor.left_inds
        L_inds = list(left_inds)
        R_inds = [ix for ix in tensor.inds if ix not in L_inds]



        LR_inds = L_inds + R_inds
        tensor.transpose_(*LR_inds)
        # fuse this tensor into a matrix and 'isometrize' it
        x = tensor.to_dense(L_inds, R_inds)
        (u,s,vh)=qu.svd(x, return_vecs=True)
        #print (x.shape, u.shape, vh.shape)
        #print (x, u@ np.diag(s) @ vh)
        #print (s)
        edata=u @ vh
        #print (np.linalg.norm(x-edata))

        edata=edata.reshape(tensor.shape)
        tensor.modify(data=edata)
        tensor.modify(left_inds=left_inds)

  return  tn


def  check_tags(tn_U, list_tags_I, list_tags_U):
  #print (   tn_U.num_tensors, len(list_tags_I)+len(list_tags_U)+tn_U.Lx*tn_U.Lz*tn_U.Ly    )
  if tn_U.num_tensors != len(list_tags_I)+len(list_tags_U)+tn_U.Lx*tn_U.Lz*tn_U.Ly:
       print ("tags are not unique", tn_U.num_tensors, len(list_tags_I)+len(list_tags_U)+tn_U.Lx*tn_U.Lz*tn_U.Ly )


  map_tags=tn_U.tag_map
  for i in list_tags_I:
    if len(map_tags[i]) != 1:
       print ("tags are not unique", i,map_tags[i] )

  for i in list_tags_U:
    if len(map_tags[i]) != 1:
       print ("tags are not unique", i,map_tags[i] )





def  check_tags_2d(tn_U, list_tags_I, list_tags_U):
  #print (   tn_U.num_tensors, len(list_tags_I)+len(list_tags_U)+tn_U.Lx*tn_U.Lz*tn_U.Ly    )
  if tn_U.num_tensors != len(list_tags_I)+len(list_tags_U)+tn_U.Lx*tn_U.Ly:
       print ("tags are not unique", tn_U.num_tensors, len(list_tags_I)+len(list_tags_U)+tn_U.Lx*tn_U.Lz*tn_U.Ly )


  map_tags=tn_U.tag_map
  for i in list_tags_I:
    if len(map_tags[i]) != 1:
       print ("tags are not unique", i,map_tags[i] )

  for i in list_tags_U:
    if len(map_tags[i]) != 1:
       print ("tags are not unique", i,map_tags[i] )









########################################################################################################

def  auto_diff_mera(tn_U, list_sites,list_inter, opt, optimizer_c='L-BFGS-B', tags=[],jit_fn=True, device="cpu"):


 print ("device=",device, "jit_fn=", jit_fn, "optimize=",opt,optimizer_c, "method_norm", method_norm)

 tnopt = qtn.TNOptimizer(
        tn_U,                          # the initial TN
        loss_fn=energy_f,                         # the loss function
        norm_fn=norm_f,                         # this is the function that 'prepares'/constrains the tn                 
        shared_tags=[],
        tags=tags,
        loss_constants={'list_sites': list_sites, 'list_inter': list_inter },  # additional tensor/tn kwargs
        loss_kwargs={'optimize': opt},
        autodiff_backend='torch',
        device=device,
        optimizer=optimizer_c, 
        jit_fn=jit_fn 
        )

 return tnopt

def   norm_f(tn_U):
    return tn_U.unitize_(method=method_norm, allow_no_left_inds=True)


def local_expectation_mera(tn_U, list_sites, list_inter, i, optimize="auto-hq"):
 where=list_sites[i]
 tags = [tn_U.site_tag(*coo) for coo in where]
 tn_ij = tn_U.select(tags, which='any')
 tn_ij_G=tn_ij.gate(list_inter[i], where)
 tn_ij_ex = (tn_ij_G & tn_ij.H)
 tn_ij_ex.rank_simplify_()
 return tn_ij_ex.contract(all, optimize=optimize) 

def energy_f(tn_U, list_sites, list_inter, **kwargs):
    return sum(
        local_expectation_mera(tn_U, list_sites, list_inter, iter, **kwargs)
        for iter in range(len(list_sites))
    )




def Mag_calc(tn_U, optimize,data_type="float64"):

  Z = qu.pauli('Z',dtype=data_type) * (0.5)
  X = qu.pauli('X',dtype=data_type) * (0.5)
  Y=np.array([[0, -1],[1,0]]) * (0.5)

  N_x=tn_U.Lx
  N_y=tn_U.Ly

  X=X.astype(data_type)
  Z=Z.astype(data_type)
  Y=Y.astype(data_type)

      
  list_sites=[]
  list_inter=[X,Y,Z]
   

  for i in range(N_x): 
    for j in range(N_y): 
          list_sites.append(  ( i,j)   )

  val_count=0
  results=[]      
  for coor1 in   list_sites:
   for coor2 in   list_sites:
      rx_i, ry_i = coor1
      rx_j, ry_j = coor2
      ii=rx_i*N_y+ry_i
      ii_=rx_j*N_y+ry_j
      #print (coor1, coor2,coor2 not in coor_tempo)
      if rx_i !=rx_j or ry_i!=ry_j :
         if ii_<ii:
                  val_count+=1
                  dis_val=(exp(1j * pi * (rx_i-rx_j) )*exp(1j * pi * (ry_i-ry_j) )).real
                  print (coor1,coor2, ii, ii_,dis_val,val_count , len(results) )
                  tags = [tn_U.site_tag(*coor1)]+[tn_U.site_tag(*coor2)]
                  tn_ij = tn_U.select(tags, which='any')
                  res=0
                  for count,ele in enumerate(list_inter):
                        #print ("count",count)
                        tn_ij_X=tn_ij.gate(ele, (coor1,) )
                        tn_ij_XX=tn_ij_X.gate(ele, (coor2,))
                        tn_ij_exX = ( tn_ij.H &  tn_ij_XX)
                        tn_ij_exX.rank_simplify_()
                        if count != 1:
                           res+=tn_ij_exX.contract(all, optimize=optimize)
                        else:
                           res-=tn_ij_exX.contract(all, optimize=optimize)

                  print (res, dis_val)
                  results.append(res*dis_val)
  return sum (results)/len(results)


##########################################################################################################
def   norm_f_tree(tn_U):
    return tn_U.unitize_(method=method_norm, allow_no_left_inds=True)


def local_expectation_mera_tree(tn_U, list_sites, list_inter, i, optimize="auto-hq"):
    where=list_sites[i]
    #print (optimize)
    coo1,coo2=where
    tags = [tn_U.site_tag(*coo) for coo in where]
    tn_ij = tn_U.select(tags, which='any')
    res=0
    for count,ele in enumerate(list_inter):
        tn_ij_X=tn_ij.gate(ele, (coo1,) )
        tn_ij_XX=tn_ij_X.gate(ele, (coo2,))

        tn_ij_exX = ( tn_ij.H &  tn_ij_XX)
        #tn_ij_exX.rank_simplify_()
        if count != 2:
          res+=tn_ij_exX.contract(all, optimize=optimize)
        else:
          res-=tn_ij_exX.contract(all, optimize=optimize)
 
    return res 

def energy_f_tree(tn_U, list_sites, list_inter, **kwargs):
    return sum(
        local_expectation_mera_tree(tn_U, list_sites, list_inter, iter, **kwargs)
        for iter in range(len(list_sites))
    )

def  auto_diff_tree(tn_U, list_sites,list_inter, opt, optimizer_c='L-BFGS-B', tags=[],jit_fn=True, device="cpu"):

 print ("device=",device, "jit_fn=", jit_fn, "optimize=",opt, optimizer_c ,"method_norm", method_norm)

 tnopt = qtn.TNOptimizer(
        tn_U,                          # the initial TN
        loss_fn=energy_f_tree,                         # the loss function
        norm_fn=norm_f_tree,                         # this is the function that 'prepares'/constrains the tn                 
        shared_tags=[],
        tags=tags,
        loss_constants={'list_sites': list_sites, 'list_inter': list_inter },  # additional tensor/tn kwargs
        loss_kwargs={ 'optimize': opt},
        autodiff_backend='torch',
        device='cpu',
        optimizer=optimizer_c,
         jit_fn=True )

 return tnopt












def  Tn_mera_build_3d(chi=4,data_type='float64', dist_type="uniform", phys_dim=2):

  data_type  = data_type
  num_layers = 2
  N_x = 4*1
  N_y = 4*1
  N_z = 4*1
  tn_U = TN3DUni.empty( N_x, N_y, N_z, phys_dim=phys_dim )
  chi = chi

  list_sites, list_inter=Heis_local_Ham_open_3D( N_x,N_y,N_z,data_type=data_type,phys_dim=phys_dim )
  #list_sites, list_inter=Heis_local_Ham_cycle_3D( N_x,N_y,N_z,data_type=data_type,phys_dim=phys_dim )

  print ( "N_x, N_y, N_z", N_x, N_y, N_z, "chi", chi )
  list_tags_U=[]
  list_tags_I=[]
  label_listU=[0]
  shared_U=[]
  shared_I=[]
  list_scale=[]
  num_layer=[0,0]     #U, I
  label_list=[0]
######################


  Iso_44_11_3D( tn_U, N_x, N_y, N_z, chi, num_layer, list_tags_U, list_tags_I, shared_I, shared_U, list_scale,
  scale=0,
  uni_h="off",
  uni_top="off", 
  uni_h_Iso="off",
  uni_h_Iso_top="off",
  uni_h_Iso_top_top="off",
  last_bond="on", 
  cycle="off", 
  Iso_apply="on",
  data_type=data_type, 
  seed_val=10,
  seed_val_u=0,
  dist_type=dist_type)

#   Iso_44_11_3D( tn_U, N_x//4, N_y//4, N_z//4, chi, num_layer, list_tags_U, list_tags_I, shared_I, shared_U, list_scale,
#   scale=0,
#   uni_h="on",
#   uni_top="off", 
#   uni_h_Iso="off",
#   uni_h_Iso_top="off",
#   uni_h_Iso_top_top="off",
#   last_bond="on", 
#   cycle="off", 
#   Iso_apply="on",
#   data_type=data_type, 
#   seed_val=10,
#   seed_val_u=0,
#   dist_type=dist_type)




#   Iso_33_11_3D( tn_U, N_x//4, N_y//4, N_z//4, chi, num_layer, list_tags_U, list_tags_I, shared_I, shared_U, list_scale,
#   scale=1,
#   uni_h="on",
#   uni_h_Iso="off",
#   uni_h_Iso_top="off",
#   last_bond="off", 
#   cycle="off", 
#   Iso_apply="on",
#   data_type=data_type, 
#   seed_val=10,
#   seed_val_u=0,
#   dist_type=dist_type)





#  uni_xy_22_3D_full(tn_U, N_x//4, N_y//4,N_z//4,num_layer,list_tags_U,chi,list_scale,1, seed_val=0, 
#                      cycle="off",dist_type=dist_type)
# #   uni_xy_22_3D_sparse(tn_U, N_x//4, N_y//4,N_z//4,num_layer,list_tags_U,chi,list_scale,scale, seed_val=0, 
# #                      cycle="off",dist_type=dist_type)
#  Iso_22_3D(tn_U, N_x//4, N_y//4,N_z//4,num_layer,list_tags_I,chi,list_scale,1,seed_val=10, 
#                      last_bond="on",dist_type=dist_type,data_type=data_type)




  check_tags(tn_U, list_tags_I, list_tags_U)
  list_scale=eliminate_dupl(list_scale)
  #tn_U.gauge_all_canonize_(max_iterations=5)
  tn_U.unitize_(method=method_norm, allow_no_left_inds=True)
  return tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale







def Tn_mera_build(chi=6,data_type='float64',dist_type="uniform",phys_dim=2):

  data_type=data_type
  num_layers = 2
  N_x=4**num_layers
  N_y=4**num_layers

  N_x=8
  N_y=8

  #N_x=4*4*2
  #N_y=4*4*2


  tn_U = TN2DUni.empty(N_x, N_y, phys_dim=phys_dim,data_type=data_type)
  list_sites, list_inter = Heis_local_Ham_open(N_x,N_y,data_type=data_type,phys_dim=phys_dim)
  #list_sites, list_inter = Heis_local_Ham_open_long(N_x,N_y,data_type=data_type,phys_dim=phys_dim)

  chi = chi


  print ("N_x, N_y", N_x, N_y, "chi", chi)
  list_tags_U=[]
  list_tags_I=[]
  label_listU=[0]
  shared_U=[]
  shared_I=[]
  list_scale=[]
  num_layer=[0,0]     #U, I
  label_list=[0]

######################
  mps_build(tn_U, N_x, N_y,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,
  scale=0,
  uni_h="on",
  uni_h_full="off",
  uni_h_Iso="off",
  cycle="off",
  data_type=data_type, 
  seed_val=120,
  seed_val_u=10,
  dist_type=dist_type)





#   Iso_44_11(tn_U, N_x, N_y,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,
#    scale=0,
#    uni_h="off",
#    uni_top="off",
#    uni_h_full="off",
#    uni_h_Iso="off",
#    last_bond="off",
#    cycle="off",
#    data_type=data_type,
#    Iso_apply="off", 
#    Iso_1=True, 
#    Iso_2=True,
#    seed_val=120,
#    seed_val_u=0,
#    dist_type=dist_type)

#   Iso_44_11(tn_U, N_x//4, N_y//4,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,
#   scale=1,
#   uni_h="off",           #sparse Horizontal and vertical
#   uni_top="off",
#   uni_h_full="off",
#   uni_h_Iso="off",
#   last_bond="off",
#   cycle="off",
#   data_type=data_type,
#   Iso_apply="on", 
#   Iso_1=True, 
#   Iso_2=True,
#   seed_val=10,
#   seed_val_u=0,
#   dist_type=dist_type)



  
  #uni_xy_22_Iso( tn_U, N_x//16, N_y//16, chi, num_layer, list_tags_U,label_listU,shared_U,list_scale, scale=2,seed_val=10, cycle="off", data_type=data_type,dist_type=dist_type)
  #Iso_22(tn_U, N_x//4, N_y//4,chi,num_layer, list_tags_I,label_list,shared_I,list_scale,scale=2,seed_val=10, last_bond="on",data_type=data_type, Iso=True,dist_type=dist_type)


########################
  #Iso_33_11(tn_U, N_x, N_y,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,
  # scale=0,
  # uni_h="on",
  # uni_h_full="off",
  # last_bond="off",
  # cycle="off",
  # data_type=data_type, 
  # Iso=True,
  # Iso_apply="on",dist_type=dist_type)

#   Iso_33_11(tn_U, N_x//3, N_y//3,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,
#   scale=1,
#   uni_h="off",
#   uni_h_full="off",
#   last_bond="on",
#   cycle="off",
#   data_type=data_type, 
#   Iso=True,
#   Iso_apply="on",dist_type=dist_type)


  #Iso_22(tn_U, N_x//3, N_y//3,
  #    chi,num_layer, 
  #    list_tags_I,
  #    label_list,
  #    shared_I,
  #    list_scale,
  #    scale=2,
  #    seed_val=40,
  #    last_bond="on",
  #    data_type=data_type,
  #    Iso=True,dist_type=dist_type)

#  quf.Iso_33_11(tn_U, N_x//9, N_y//9,chi,
#  num_layer,list_tags_U,
#  list_tags_I,shared_I,shared_U,
#  list_scale,
#  scale=2,
#  uni_h="off",
#  uni_h_full="off",
#  last_bond="on",
#  cycle="off",
#  data_type=data_type, Iso=True,Iso_apply="on",dist_type=dist_type)

  check_tags_2d(tn_U, list_tags_I, list_tags_U)
  list_scale=eliminate_dupl(list_scale)
  #tn_U.gauge_all_canonize_(max_iterations=5)
  tn_U.unitize_(method=method_norm, allow_no_left_inds=True)
  #tn_U.astype(dtype, inplace=True)
  return tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale











def   Tn_tree_build(chi=6,data_type='float64', dist_type="uniform"):

  data_type=data_type
  num_layers = 2
  N_x=4**num_layers
  N_y=4**num_layers

  N_x=3*3
  N_y=3*3

  N_x=4*4*2
  N_y=4*4*2


  tn_U = TN2DUni.empty(N_x, N_y, phys_dim=2,data_type=data_type)
  list_sites, list_inter = Heis_local_Ham_open_tree(N_x,N_y,data_type=data_type)

  chi = chi


  print ("N_x, N_y", N_x, N_y, "chi", chi)
  list_tags_U=[]
  list_tags_I=[]
  shared_U=[]
  shared_I=[]
  list_scale=[]
  num_layer=[0,0]     #U, I
  label_list=[0]

######################

  Iso_44_11(tn_U, N_x, N_y,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,
  scale=0,
  uni_h="off",
  uni_top="off",
  uni_h_full="off",
  last_bond="off",
  cycle="off",
  data_type=data_type,
  Iso_apply="on", 
  Iso_1=True, 
  Iso_2=True,
  seed_val=120,
  dist_type=dist_type)

  Iso_44_11(tn_U, N_x//4, N_y//4,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,
  scale=1,
  uni_h="off",
  uni_top="off",
  uni_h_full="off",
  last_bond="off",
  cycle="off",
  data_type=data_type,
  Iso_apply="on", 
  Iso_1=True, 
  Iso_2=True,
 dist_type=dist_type)

  Iso_22(tn_U, N_x//16, N_y//16,chi,num_layer, list_tags_I,label_list,shared_I,list_scale,scale=2,seed_val=10, last_bond="on",data_type=data_type, Iso=True,dist_type=dist_type)


########################
#   Iso_33_11(tn_U, N_x, N_y,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,
#   scale=0,
#   uni_h="off",
#   uni_h_full="off",
#   last_bond="off",
#   cycle="off",
#   data_type=data_type, 
#   Iso=True,
#   Iso_apply="on",dist_type=dist_type)

#   Iso_33_11(tn_U, N_x//3, N_y//3,chi,num_layer,list_tags_U,list_tags_I,shared_I,shared_U,list_scale,
#   scale=1,
#   uni_h="off",
#   uni_h_full="off",
#   last_bond="on",
#   cycle="off",
#   data_type=data_type, 
#   Iso=True,
#   Iso_apply="on",dist_type=dist_type)


#  Iso_22(tn_U, N_x//3, N_y//3,
#  chi,num_layer, 
#  list_tags_I,
#  label_list,
#  shared_I,
#  list_scale,
#  scale=2,
#  seed_val=40,
#  last_bond="on",
#  data_type=data_type,
#  Iso=True,dist_type=dist_type)

#  quf.Iso_33_11(tn_U, N_x//9, N_y//9,chi,
#  num_layer,list_tags_U,
#  list_tags_I,shared_I,shared_U,
#  list_scale,
#  scale=2,
#  uni_h="off",
#  uni_h_full="off",
#  last_bond="on",
#  cycle="off",
#  data_type=data_type, Iso=True,Iso_apply="on",dist_type=dist_type)

  check_tags(tn_U, list_tags_I, list_tags_U)
  list_scale=eliminate_dupl(list_scale)

  return tn_U,list_sites, list_inter,list_tags_I, list_tags_U,list_scale













