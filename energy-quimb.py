from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
from redis.client import timestamp_to_datetime
import quf
import mera2d
import mera3d
import dmrg3d
import tree2d
import contractinfo
import miniatureTN
import matplotlib.pyplot as plt
from numpy import linalg as LA
import os
import multiprocessing
import ray
ray.shutdown()

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path=dir_path+"/temp/"
ray.init(_temp_dir=dir_path, 
         #num_cpus=4,
         #num_gpus=2,
         #address='auto'
         )

#ray start --head --num-cpus=12  --num-gpus=2



os.environ['CUDA_VISIBLE_DEVICES'] = "3"
NUM_THREADS="1"
NUM_THREADS_OMP="1"
os.environ['NUMBA_NUM_THREADS']=NUM_THREADS
os.environ['OMP_NUM_THREADS']=NUM_THREADS_OMP
os.environ["MKL_NUM_THREADS"] = NUM_THREADS_OMP
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS_OMP
print ( "NUMBA_NUM_THREADS", os.environ['NUMBA_NUM_THREADS'], "OMP_NUM_THREADS", os.environ['OMP_NUM_THREADS'] )
print( "cpus", multiprocessing.cpu_count(), "cpus_used_actual", len(os.sched_getaffinity(0)) )




if __name__ == '__main__':
 #mera2d.mera_tn_2d( )
 miniatureTN.miniature_tn_( )
 #tree2d.tree_tn_2d( )
 #mera3d.mera_tn_3d( )
 #contractinfo.cont_tn_2d()
 #dmrg3d.dmrg()
 #contractinfo.cont_tn_2d()
 ray.shutdown()
