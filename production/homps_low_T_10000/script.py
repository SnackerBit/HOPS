import numpy as np
import sys
sys.path.append("../../")
from src.util import operators
from src.homps import homps
from src.util import noise_generator
from src.util import debye_spectral_density

# Parameters for the spin-boson model
epsilon = 2.0
delta = -2.0
# Parameters for the Debye spectral density
beta = 50
T = 1/beta
gamma = 5.0
eta = 0.5
# operators
sigma_x, sigma_z, eye = operators.generate_physical_operators()
L = sigma_z
h = operators.generate_spin_boson_hamiltonian(delta=delta, epsilon=epsilon)
# time window
duration = 30

N_steps = 1500
N_trunc = 9
N_terms = 13

g, w = debye_spectral_density.get_debye_coefficients(N_terms, T, gamma, eta, mode='pade')
options = {
    'linear' : False,
    'use_noise' : True,
    'chi_max' : 10,
    'eps' : 1e-3,
    'method' : 'RK4',
    'rescale_aux' : True,
}

my_homps = homps.HOMPS_Engine(g, w, h, L, duration, N_steps, N_trunc, options)
my_homps.compute_realizations(N_samples=10000, start=8000, data_path="data/psi")
