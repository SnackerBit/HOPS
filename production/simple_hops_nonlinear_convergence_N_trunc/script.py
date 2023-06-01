import numpy as np
import sys
sys.path.append("../../")
from src.util import operators
from src.hops import hops

# parameters for the bath correlation functions
g = np.array([2])
w = np.array([0.5+2j])
# time
duration = 50
# operators
sigma_x, sigma_z, eye = operators.generate_physical_operators()
h = operators.generate_spin_boson_hamiltonian()
L = sigma_z

N_steps = 1000
N_trunc_list = [16]
options = {
    'linear' : False,
    'use_noise' : True,
}
for N_trunc in N_trunc_list:
    my_hops = hops.HOPS_Engine_Simple(g, w, h, L, duration, N_steps, N_trunc, options)
    my_hops.compute_realizations(N_samples=10000, start=8800, data_path="data/psi_N_trunc_"+str(N_trunc)+"_")
