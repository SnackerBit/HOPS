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
beta = 50
T = 1/beta
gamma = 5.0
eta = 0.5
# operators
sigma_x, sigma_z, eye = operators.generate_physical_operators()
L = sigma_z
h = operators.generate_spin_boson_hamiltonian(delta=delta, epsilon=epsilon)
# time window
duration = 5

N_steps = 10000
taus = np.linspace(0, duration, N_steps)[1:]
np.save("data/taus", taus)

J = lambda w : debye_spectral_density.debye_spectral_density(w, eta, gamma)

w_cut = 1000
N = 100000
dw = w_cut/N

alphas_compare = np.array([bath_correlation_function.alpha_sum(tau, J, beta, N, dw) for tau in taus])

np.save("data/alphas_compare", alphas_compare)

alphas_matsubara = []
N_terms_matsubara = [50, 100, 1000]
for N_terms in N_terms_matsubara:
    g, w = debye_spectral_density.get_debye_coefficients(N_terms, T, gamma, eta, mode='matsubara')
    alphas = bath_correlation_function.alpha(taus, g, w)
    alphas_matsubara.append(alphas)
    
np.save("data/alphas_matsubara", alphas_matsubara)

alphas_pade = []
N_terms_pade = [5, 13, 30]
for N_terms in N_terms_pade:
    g, w = debye_spectral_density.get_debye_coefficients(N_terms, T, gamma, eta, mode='pade')
    alphas = bath_correlation_function.alpha(taus, g, w)
    alphas_pade.append(alphas)
    
np.save("data/alphas_pade", alphas_pade)