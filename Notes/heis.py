# ****************************************************************************
#
# ALPS Project: Algorithms and Libraries for Physics Simulations
#
# ALPS Libraries
#
# Copyright (C) 2010 by Brigitte Surer
#
# This software is part of the ALPS libraries, published under the ALPS
# Library License; you can use, redistribute it and/or modify it under
# the terms of the license, either version 1 or (at your option) any later
# version.
#
# You should have received a copy of the ALPS Library License along with
# the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
# available from http://alps.comp-phys.org/.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
# SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
# FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# ****************************************************************************
import matplotlib
matplotlib.use('Agg')
import pyalps
import matplotlib.pyplot as plt
import pyalps.plot
import numpy as np
import pyalps.fit_wrapper as fw
from math import sqrt

# prepare the input parameters
parms = []
# for j1 in [0.0]:
# for j1 in [1e-5,0.25]:
    # for T in [1e-1,2e-1,5e-1,7e-1,1,2,5,7,10]:
    # for T in [1e-2,1e-1,2e-1,5e-2,7e-2,1,2,5,7,10]:
# for j1 in [1e-5, 0.25, 0.5, 0.75, 1.0]:
for T in [0.003]:
    parms.append(
        {
          'LATTICE'        : "inhomogeneous simple cubic lattice",
          'local_S'        : 0.5,
          'ALGORITHM'      : 'loop',
          'SEED'           : 0,
          'T'              : T,
          'J'              : 1,
          # 'J1'             : j1,
          'THERMALIZATION' : 5000,
          'SWEEPS'         : 500,
          'MODEL'          : "spin",
          'L'              : 4,
          'W'              : 4,
          'H'              : 4,
        }
    )

# write the input file and run the simulation
input_file = pyalps.writeInputFiles('heis3d', parms)
# pyalps.runApplication('dirloop_sse',input_file)
pyalps.runApplication('loop', input_file)

# data = pyalps.loadMeasurements(pyalps.getResultFiles(prefix='heis2d'))
# measurements = { item.props['observable'] for item in pyalps.flatten(data) }
# print measurements
data = pyalps.loadMeasurements(
    pyalps.getResultFiles(pattern='heis3d.task*.out.h5'),
    ['T', 'Sign', 'Energy']
)
sign = pyalps.collectXY(
    data, x='T', y='Sign',
    # foreach=['J1']
)
energy = pyalps.collectXY(
    data, x='T', y='Energy',
    # foreach=['J1']
)

print(data)
print()
print(energy)

# Plot
plt.figure()
pyalps.plot.plot(sign)
plt.xscale("log")
plt.xlabel(r'$T$')
plt.ylabel(r'Sign')
plt.legend()
plt.show()
plt.savefig('result-sign.pdf')

pyalps.plot.plot(energy)
plt.xscale("log")
plt.xlabel(r'$T$')
plt.ylabel(r'Energy')
plt.legend()
plt.show()
plt.savefig('result-energy.pdf')
