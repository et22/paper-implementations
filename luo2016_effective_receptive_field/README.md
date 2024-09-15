# Luo et al. 2016 Understanding the Effective Receptive Field in Deep Convolutional Neural Networks 

### Overview
Luo and colleagues examine the effective receptive field (ERF) - the region of space that impacts a neuron's response - of neurons in deep convolutional neural networks. They show that ERFs in CNNs tend to be gaussian and substantially smaller than the maximum theoretical RF size. Here, I have implemented their method for computing the effective receptive field to reproduce a results from the paper and to examine the effective receptive field sizes of all layers in ResNet18 and ResNet50 models. 

### Results

<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/luo2016_effective_receptive_field/figures/figure1.png" alt="figure1" width="1000"/>
</p>

*Figure 1.* Effective receptive fields (ERFs) for linear CNNs with uniform weights, linear CNNs with random weights, CNNs with random weights and ReLU activation function. ERFs are shown for four networks with 5, 10, 20, and 40 layers. 

<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/luo2016_effective_receptive_field/figures/figure2.png" alt="figure2" width="330"/>
</p>

*Figure 2.* ERFs for layer3.0.conv1 in a ResNet18 model with randomly initialized weights (left) or pretrained on object classification (right). 

![Figure 3](https://github.com/et22/paper-implementations/blob/main/luo2016_effective_receptive_field/figures/figure3.png)
*Figure 3.* ERFs for all convolutional layers in a ResNet18 model pretrained on object classification. 

![Figure 4](https://github.com/et22/paper-implementations/blob/main/luo2016_effective_receptive_field/figures/figure4.png)
*Figure 4.* ERFs for all convolutional layers in a ResNet50 model pretrained on object classification. 

### Reproducing the results 
Results can be reproduced by running the Jupyter notebook `effective_receptive_field.ipynb`. 

### Acknowledgements
Referenced the original paper, and used Gemini to assist with writing code. 