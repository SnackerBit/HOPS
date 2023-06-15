import numpy as np
import sys
sys.path.append("../../")
from src.util import operators
from src.homps import homps
from src.hops import hops
from src.util import noise_generator
from src.util import debye_spectral_density

# Parameters for the spin-boson model
epsilon = 2.0
delta = -2.0
# Parameters for the Debye spectral density
beta = 0.5
T = 1/beta
gamma = 0.25
eta = 0.5
# operators
sigma_x, sigma_z, eye = operators.generate_physical_operators()
L = sigma_z
h = operators.generate_spin_boson_hamiltonian(delta=delta, epsilon=epsilon)
# time window
duration = 30

N_steps = 500
N_trunc = 40
N_terms = 1
N_samples = 10000

g, w = debye_spectral_density.get_debye_coefficients(N_terms, T, gamma, eta)

# HOPS
options = {
    'linear' : False,
    'use_noise' : True,
}
my_hops = hops.HOPS_Engine_Simple(g, w, h, sigma_z, duration, N_steps, N_trunc, options)
my_hops.compute_realizations(N_samples, data_path="data/HOPS/psi")

# HOMPS (RK4)
options = {
    'linear' : False,
    'use_noise' : True,
    'chi_max' : 10,
    'eps' : 0,
    'method' : 'RK4',
    'rescale_aux' : True,
}
my_homps = homps.HOMPS_Engine(g, w, h, sigma_z, duration, N_steps, N_trunc, options)
my_homps.compute_realizations(N_samples, data_path="data/HOMPS_RK4/psi")

# HOMPS (TDVP)
options = {
    'linear' : False,
    'use_noise' : True,
    'chi_max' : 10,
    'eps' : 0,
    'method' : 'TDVP2',
    'rescale_aux' : True,
}
my_homps = homps.HOMPS_Engine(g, w, h, sigma_z, duration, N_steps, N_trunc, options)
my_homps.compute_realizations(N_samples, data_path="data/HOMPS_TDVP/psi")
