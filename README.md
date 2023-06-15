# Hierarchy of Pure States (HOPS) and Hierarchy of Matrix Product States (HOMPS)
## Simulating the time evolution of open non-Markovian quantum systems
This repository contains implementations for the Hierarchy of Pure States [[1]](#1) and Hierarchy of Matrix Product States [[2]](#2) methods.
The methods are based on the Non-Markovian Quantum State Diffusion Equation [[3]](#3). <br />

This implementation was created as part of my bachelor's thesis [[4]](#4). In the thesis, the derivation of HOPS and HOMPS and the implementation
of both methods is explained in detail. <br />

The repository includes three folders, `src`, `test`, and `production`. In the `src` folder the actual implementation is given in the
form of python code. In the `test` folder different parts of the implementation are tested with jupyter notebooks. Finally, in the `production`
folder, the scripts that were used to produce all plots in my thesis are given.

## References
<a id="1">[1]</a> 
D. Suess, A. Eisfeld, and W. T. Strunz. “Hierarchy of Stochastic Pure States for Open Quantum System Dynamics”. In: Phys. Rev. Lett. 113 (15 2014), p. 150403 <br />
<a id="2">[2]</a> 
X. Gao, J. Ren, A. Eisfeld, and Z. Shuai. “Non-Markovian stochastic Schrödinger equation: Matrix-product-state approach to the hierarchy of pure states”. In: Phys. Rev. A 105 (3 2022), p. L030202. <br />
<a id="3">[3]</a> 
L. Diósi, N. Gisin, and W. T. Strunz. “Non-Markovian quantum state diffusion”. In: Phys. Rev. A 58 (3 1998), pp. 1699–1712 <br />
<a id="4">[4]</a> 
Benjamin Sappler, "Benchmarking a Tensor Network Algorithm for the HOPS-Method to Simulate Open non-Markovian Quantum Sytems" https://github.com/SnackerBit/bachelors-thesis-computer-science <br />
