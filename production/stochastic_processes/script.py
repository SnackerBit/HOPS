import numpy as np
import sys
sys.path.append("../../")
from src.util import noise_generator
from src.util import bath_correlation_function

g = np.array([2])
w = np.array([0.5+2j])
t_start = 0
t_stop = 10
N_steps = 500
taus = np.linspace(t_start, t_stop, N_steps)
alpha = lambda tau : bath_correlation_function.alpha(tau, g, w)


def make_some_noise(N_samples):
    generator = noise_generator.ColoredNoiseGenerator_FourierFiltering(alpha, t_start, t_stop)
    generator.initialize(N_steps)
    noise = np.empty((N_samples, N_steps), dtype=complex)
    for i in range(N_samples):
        noise[i, :] = generator.sample_process()
    return noise


noise = make_some_noise(100)
np.save("data/noise_100", noise)
noise = make_some_noise(1000)
np.save("data/noise_1000", noise)
noise = make_some_noise(10000)
np.save("data/noise_10000", noise)
noise = make_some_noise(100000)
np.save("data/noise_100000", noise)
