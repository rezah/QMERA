import quimb as qu
import quimb.tensor as qtn
import cotengra as ctg
import functools
from cotengra.parallel import RayExecutor
from concurrent.futures import ProcessPoolExecutor

L = 2**8
D = 4
dtype = 'float64'
mera = qtn.MERA.rand(L, max_bond=D, dtype=dtype)
mera.unitize_(method='mgs', allow_no_left_inds=True)



H2 = qu.ham_heis(2).real.astype(dtype)
terms = {(i, (i + 1) % L): H2 for i in range(L)}

def norm_fn(mera):
    return mera.unitize(method='mgs', allow_no_left_inds=True)

def loss_i(mera, terms, where, optimize='auto-hq'):
    tags = [mera.site_tag(coo) for coo in where]
    mera_ij = mera.select(tags, 'any')

    G = terms[where]
    mera_ij_G = mera_ij.gate(terms[where], where)

    mera_ij_ex = (mera_ij_G & mera_ij.H)
    return mera_ij_ex.contract(all, optimize=optimize)

loss_fns = [
    functools.partial(loss_i, where=where)
    for where in terms
]


opt='auto-hq'
opt = ctg.ReusableHyperOptimizer(
     progbar=True,
     minimize='flops',       #{'size', 'flops', 'combo'}, what to target
     reconf_opts={}, 
     max_repeats=2**6,
     max_time=3600,
#    max_time='rate:1e6',
     parallel=True,
     #optlib='baytune',         # 'nevergrad', 'baytune', 'chocolate','random'
     directory="cash/"
 )


executor = RayExecutor()
executor = ProcessPoolExecutor()

tnopt_l = qtn.TNOptimizer(
    mera,
    loss_fn=loss_fns,
    norm_fn=norm_fn,
    loss_constants={'terms': terms},
    loss_kwargs={'optimize': opt},
    autodiff_backend='torch',
    jit_fn=True,
    executor=executor,
    device="cpu"
)



tnopt_l.optimizer = 'l-bfgs-b'  # the default
mera_opt_l = tnopt_l.optimize(10)




