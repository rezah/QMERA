import math
import quimb.tensor as qtn
import quimb as qu
import cotengra as ctg
import autoray as ar
import quf

def to_backend(x):
    import torch
    #return torch.tensor(x).cuda()
    return torch.tensor(x).cpu()
    
    
#tn = qtn.TN3D_classical_ising_partition_function(3, 3, 3, beta=0.3)

#peps = qtn.PEPS.rand(Lx=6, Ly=6,bond_dim=4, seed=10)
#tn = peps.make_norm()
#tn.flatten_()
#tn.draw(color=tn.site_tags, show_tags=False,legend=False, figsize=(4, 4))
L_L=2**6
qmera = qtn.MERA.rand(L_L, max_bond=12, dtype='float64',seed=10)
qmera=qu.load_from_disk("Store/qmera")
#qmera.unitize_()
opt='auto-hq'
ZZ = qu.pauli('Z', dtype="float64") & qu.pauli('Z',dtype="float64")
YY = qu.pauli('Y') & qu.pauli('Y')
XX = qu.pauli('X', dtype="float64") & qu.pauli('X',dtype="float64")
H2=(ZZ+XX+YY)*(1./4.)
H2=H2.astype("float64") 
#H2=(qu.pauli("I") & qu.pauli("I")).real
i=22
where=(i,(i+1)%L_L)
tags = [ qmera.site_tag(coo) for coo in where ]
mera_ij = qmera.select(tags, which='any')
mera_ij_G=mera_ij.gate(H2, where)
mera_ij_ex = (mera_ij_G & mera_ij.H)
print ( "contract", i, mera_ij_ex.contraction_width( optimize=opt),mera_ij_ex^all )
tn=mera_ij_ex


tn.apply_to_arrays(to_backend)
tn.draw(color=[f'_LAYER{i}' for i in range(7)])



opt = ctg.HyperOptimizer(
    slicing_reconf_opts={'target_size': 2**30},  
    progbar=True,
)

#parallel='ray'
tree_ex = tn.contraction_tree(opt)
ex = tree_ex.contract(tn.arrays, progbar=True)
ex=float(ex)
print (ex)


copt = ctg.ReusableHyperCompressedOptimizer(
    #methods=[],
    chi=16,
    minimize='peak-compressed',
    max_repeats=400,
    progbar=True,
    directory="cashlab/"
)
#parallel='ray'


tree = tn.contraction_tree(copt)

tree.plot_tent(order=tree.surface_order)

chis = [4, 8, 12,16,18]

[ 
    (chi, math.log2(tree.peak_size_compressed(chi=chi)))
    for chi in chis
]

tn.gauge_all_simple_()

y_none = []

for chi in chis:
    y_none.append(
        tn.contract_compressed(
            copt, 
            max_bond=chi,
            cutoff=0.0,
        )
    )
#print (y_none)
#y_none=[i^all for i in y_none]

print ( y_none )
y_err_none = [abs(1 - y / ex).item() for y in y_none]
y_err_none


y_basic = []

for chi in chis:
    y_basic.append(
        tn.contract_compressed(
            copt, 
            max_bond=chi,
            cutoff=0.0,
            canonize_distance=2,
            canonize_after_distance=2,
            compress_late=True,
            gauge_boundary_only=False,
        )
    )
    
#y_basic=[i^all for i in y_basic]

#print (y_basic[0])
y_err_basic = [abs(1 - (y / ex) ).item() for y in y_basic]
y_err_basic




y_full = []
chis = [12,16,20,40,60]
for chi in chis:
    y_full.append(
        tn.contract_compressed(
            copt, 
            max_bond=chi, 
            compress_opts=dict(
                mode='full-bond',
                # how we form the bond environment
                env_method='contract_compressed',
                # extra options to that function, which here itself is a compressed_contract
                contract_compressed_opts=dict(
                    optimize=copt,
                    canonize_distance=2,
                    canonize_after_distance=2,
                    compress_late=True
                ),
            ),
        )
    )


y_err_full = [abs(1 - y / ex).item() for y in y_full]
y_err_full



import matplotlib.pyplot as plt
plt.plot(chis, y_err_none, marker='.', label='none')
plt.plot(chis, y_err_basic, marker='.', label='basic')
plt.plot(chis, y_err_full, marker='.', label='full-bond')
plt.yscale('log')
plt.legend()
plt.show()



















