# Rajan & Abbott 2006 Eigenvalue Spectra of Random Matrices for Neural Networks

### Overview
This repo contains an unoffical Python implementation of analytical results and numerical simulations of the distributions of eigenvalues for  different random matrices from the paper [*Eigenvalue Spectra of Random Matrices for Neural Networks*](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.188104) by Kanaka Rajan and Larry Abbott. 

### Results
Girko's circle law, a key result in random matrix theory, states that the distribution of eigenvalues of a set of n x n matrices whose entries are drawn independently from a normal distribution are uniformly distributed on a disk in the complex plane in the limit of large n (e.g., see [Weisstein](https://mathworld.wolfram.com/GirkosCircularLaw.html)).

Rajan and Abbot extend Girko's circle law to synaptic connectivity matrices for biological neural networks. These matrices have entries that are drawn not from the standard normal distribution but from one of two normal distributions with different means and variances (corresponding to excitatory and inhibitory neurons). If matrix entries are drawn from two distributions with different means but the same variance, the eigenvalues of the matrix are uniformly distributed on the unit disk with a few outliers outside the unit disk (**Figure 1, left**). These outliers can be eliminated by imposing the constraint that the sum of the synaptic inputs to each neuron is zero (**Figure 1, right**).  If matrix entries are instead drawn from two distributions with different variances, the distribution of eigenvalues concentrates towards the center of the unit disk (**Figure 2, inset**).

![Eigenvalue distribution with matched variance](https://github.com/et22/paper-implementations/blob/main/rajan2006_eigenvalue_spectra/figure1.png)
*Figure 1.* Distribution of eigenvalues for random connectivity matrices with different mean synaptic weights for excitatory and inhibitory neurons but the same variance (left). Same as left panel but with the additional constraint that excitatory and inhibitory inputs to each neuron are balanced (i.e., rows in the connectivity matrix sum to zero).

![Eigenvalue density with different variance](https://github.com/et22/paper-implementations/blob/main/rajan2006_eigenvalue_spectra/figure2.png)
*Figure 2.* Distribution of eigenvalues for random connectivity matrices with different variances for excitatory and inhibitory synaptic weights. Eigenvalue density ($\rho$) is plotted as a function of distance from the origin $|\omega|$ for simulations with different fractions of excitatory neurons $f$ (left). Eigenvalue density as a function of distance for different excitatory synaptic weight variances ($\frac{1}{N*\alpha}$). 

### Reproducing the results 
Results can be reproduced by running the Jupyter notebook `eigenvalue_spectra.ipynb`. Simulation parameters are enumerated in the notebook and were selected to match the original paper as closely as possible. 

Numerical results are from a simulation of a single $n$ x $n$ random matrix ($n = 1000$). This was sufficient to replicate the main results from the paper, but increasing the number or size of matrices in numerical simulations would like lead to closer match with theory. 

### Acknowledgements
Referenced the statement of Girko's circular law [here](https://mathworld.wolfram.com/GirkosCircularLaw.html) along with the original paper.