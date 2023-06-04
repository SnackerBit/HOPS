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

g, w = debye_spectral_density.get_debye_coefficients(N_terms, T, gamma, eta)

options = {
    'linear' : False,
    'use_noise' : True,
    'chi_max' : 10,
    'eps' : 0,
    'method' : 'RK4',
    'rescale_aux' : True,
}

my_homps = homps.HOMPS_Engine(g, w, h, L, duration, N_steps, N_trunc, options)
my_homps.compute_realizations(N_samples=1000, data_path="data/psi")
