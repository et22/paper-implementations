# Sutton 1988 Learning to Predict by the Methods of Temporal Differences 

### Overview
This repo contains an unoffical Python implementation of some results from the paper [*Learning to Predict by the Methods of Temporal Differences*](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf) by Richard Sutton. 

### Results
Sutton proposed the TD($\lambda$) algorithm, a learning rule that updates predictions based on the "difference between temporally successive predictions" rather than the "difference between predicted and actual outcomes". The paper applies the TD($\lambda$) algorithm to the simple problem of predicting the expected value of the states in a bounded random walk. The plots below illustrate how varying $\lambda$ and $\alpha$ affect the root mean squared error (RMSE) in the random walk.

<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/sutton1988_td_lambda/figures/figure3.png" alt="" width="400"/>
</p>

*Figure 1.* Average RMSE with TD($\lambda$) trained on repeated presentations of random walks as a function of $\lambda$.  

<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/sutton1988_td_lambda/figures/figure4.png" alt="" width="400"/>
</p>

*Figure 2.* Average RMSE with TD($\lambda$) trained on a single presentation of 10 random walks as a function of $\alpha$ with different values of $\lambda$.   

<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/sutton1988_td_lambda/figures/figure5.png" alt="" width="400"/>
</p>

*Figure 3.* Average RMSE with TD($\lambda$) trained on a single presentation of 10 random walks as a function of $\lambda$ for the best $\alpha$ value at each $\lambda$.  

### Reproducing the results 
Results can be reproduced by running `python run.py`.

### Acknowledgements
Referenced the original paper. 