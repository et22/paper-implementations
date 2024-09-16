# Olah et al. 2017 Feature Visualization

### Overview
This repo contains PyTorch code to visualize features at different layers of a ResNet50 using regularized gradient ascent inspired by the paper/blog post [*Feature Visualization*](https://distill.pub/2017/feature-visualization/#enemy-of-feature-vis) by Olah and colleagues. 

### Results
Gradient ascent with blur + image transformations as regularizers enables learning images that strongly activate a given channel. For example, by doing, gradient descent on channel 373  in the final fully connected layer, we learn an image containing rhesus macaques and rhesus macaque parts (**Figure 1**). By applying this method to a few channels in each layer, we can begin to understand how the network successively builds up more complex representations. The first convolutional layer contains separate color detectors and edge detectors/gabor filters, the second includes joint color/edge detectors, the third captures more abstract shapes, the fourth more detailed textures, and the final fully connected layer contains shapes and textures associated with entire animals - like a lion or monkey (**Figure 2**).


<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/olah2017_feature_visualization/figures/figure1.png" alt="figure1" width="1000"/>
</p>

*Figure 1.* Evolution of image over the course of gradient ascent on the 'macaque' neuron in the fully connected layer of ResNet50. 

<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/olah2017_feature_visualization/figures/figure2.png" alt="figure2" width="1000"/>
</p>

*Figure 2.* Optimized images for 2 channels following selected convolutional and fully connected layers in ResNet50. 

### Reproducing the results 
Results can be reproduced by running the Jupyter notebook `feature_visualization.ipynb`. 

### Acknowledgements
Referenced the original paper, and used Gemini to assist with writing code. 