from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import quf
import mera2d
import mera3d
import tree2d
import contractinfo
import matplotlib.pyplot as plt
from numpy import linalg as LA
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
#pip install --no-deps -U -e .



if __name__ == '__main__':
 #mera2d.mera_tn_2d( )
 #tree2d.tree_tn_2d( )
 mera3d.mera_tn_3d( )
 #contractinfo.cont_tn_2d()
