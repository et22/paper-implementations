# Defazio & Mishchenko 2023 Learning-Rate-Free Learning by D-Adaptation

### Overview
This repo contains an unoffical Python implementation of some of the algorithms for learning-rate-free learning from the paper [*Learning-Rate-Free Learning by D-Adaptation*](https://proceedings.mlr.press/v202/defazio23a) by Aaron Defazio and Konstantin Mishchenko. 

### Results
Defazio & Mischenko propose an approach to automatically set learning rate that acheives the same rate of convergence asymptotically as the optimal learning rate identified via a hyperparameter search. This approach is based on estimating the quantity $D=|x_0 - x_*|$, the distance from the initial point $x_0$ to the point $x_*$ that minimizes the function. 

Here, I have implemented Algorithm 1, Dual Averaging with D-Adaptation, from the paper. I then applied it to the toy problem of minimizing the absolute value function to produce Figure 1 (**Figure 1**).  

![Figure 1](https://github.com/et22/paper-implementations/blob/main/defazio2023_d_adaptation/figure1.png)
*Figure 1.* Evolution of $x$ (left) and estimate of $d$ (right) obtained by applying dual averaging with d-adaptation to the absolute value function with $x_0=1.0$. 

### Reproducing the results 
Results can be reproduced by running the Jupyter notebook `d_adaptation.ipynb`. 

### Acknowledgements
Referenced the original paper.