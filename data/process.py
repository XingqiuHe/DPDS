import os
import numpy as np
from scipy.io import loadmat, savemat

alglist = ['dpds', 'coo', 'dpl', 'lpo']
nlist = list(range(10, 21))

A_scale = {}
E_scale = {}
A_energy = {}
E_energy = {}

for alg in alglist:
    A_scale[alg] = []
    E_scale[alg] = []
    A_energy[alg] = []
    E_energy[alg] = []

for alg in alglist:
    for n in nlist:
        filename = alg + '_scale_n' + str(n) + '_lam5000.mat'
        data = loadmat(filename)
        A_scale[alg].append(np.mean(data['A']))
        E_scale[alg].append(np.mean(data['E']))

print(A_scale)
print(E_scale)

savedata = {'A': A_scale, 'E': E_scale}
savemat('scale.mat', savedata)


for alg in alglist:
    for n in nlist:
        filename = alg + '_energy' + str(n) + '_n15_lam5000.mat'
        data = loadmat(filename)
        A_energy[alg].append(np.mean(data['A']))
        E_energy[alg].append(np.mean(data['E']))

print(A_energy)
print(E_energy)

savedata = {'A': A_energy, 'E': E_energy}
savemat('energy.mat', savedata)
