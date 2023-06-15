import numpy as np
import sys
sys.path.append("../../")
from src.util import operators
from src.homps import homps
from src.util import noise_generator
from src.util import debye_spectral_density
from src.util import bath_correlation_function

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

N_steps = 1000
taus = np.linspace(0, duration, N_steps)

J = lambda w : debye_spectral_density.debye_spectral_density(w, eta, gamma)

w_cut = 1000
N = 100000
dw = w_cut/N

alphas_compare = np.array([bath_correlation_function.alpha_sum(tau, J, beta, N, dw) for tau in taus])

g, w, = debye_spectral_density.get_debye_coefficients(1, T, gamma, eta)
alphas = bath_correlation_function.alpha(taus, g, w)

np.save("data/taus", taus[1:])
np.save("data/alphas_compare", alphas_compare[1:])
np.save("data/alphas", alphas[1:])