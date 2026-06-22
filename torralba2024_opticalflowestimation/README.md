# Torralba et al. 2024 Foundations of Computer Vision Chapter 48 - Optical Flow Estimation

### Overview
This repo contains an unoffical Python implementation of the simple gradient-based optic flow algorithm discussed in Chapter 48 of [*Foundations of Computer Vision*](https://visionbook.mit.edu/optical_flow.html) by Antonio Torralba, Phillip Isola, and William Freeman. 

### Results
<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_opticalflowestimation/outputs/out.gif" alt="Optic flow computed for translating squares." width="400"/>
</p>

*Figure 1.* Translating squares stimulus for computing optic flow. 

<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_opticalflowestimation/outputs/grad_frames.png" alt="Optic flow computed for translating squares." width="400"/>
</p>

*Figure 2.* Image derivatives (lx, ly, and lt).  


<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_opticalflowestimation/outputs/gradsquare_frames_postsmooth.png" alt="Optic flow computed for translating squares." width="400"/>
</p>

*Figure 3.* Smoothed higher-order terms.  


<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_opticalflowestimation/outputs/trans_squares_vid.gif" alt="Optic flow computed for translating squares." width="400"/>
</p>

*Figure 4.* Gradient-based optic flow estimation for two translating squares. 

<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_opticalflowestimation/outputs/trans_squares_vid_reliable.gif" alt="Optic flow computed for translating squares." width="400"/>
</p>

*Figure 5.* Gradient-based optic flow estimation for two translating squares in regions where the aperture-problem can be resolved. 

To reproduce, run ```python optic_flow.py```.

### References
Wrote all code from scratch with the exception of apply_sep_conv which was adapted from code written by Claude. 
