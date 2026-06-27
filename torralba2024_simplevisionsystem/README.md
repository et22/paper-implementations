# Torralba et al. 2024 Foundations of Computer Vision Chapter 2 - A Simple Vision System

### Overview
This repo contains an unoffical Python implementation of a simple image to 3D reconstruction algorithm for block world inspired by Chapter 2 of [*Foundations of Computer Vision*](https://visionbook.mit.edu/simplesystem.html) by Antonio Torralba, Phillip Isola, and William Freeman. The implementation currently relies on face segmentation and additional face-based constraints rather than strictly following the method in the book, and the resulting reconstructions are still not quite right.

### Results
I first constructed block world and took a photo from a large distance with zoom to obtain an image that can be described by parallel projection (**Figure 1, left**). However, some of the blocks in my block world were poorly constructed and lacked the well-defined edges important for the algorithm to work properly, so I relied on the block world image from chapter 2 instead (**Figure 1, right**).

<table align="center">
<tr>
<td align="center" bgcolor="white">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_simplevisionsystem/outputs/blockworld.jpeg" alt="My block world." width="300"/><br>
<i>Figure 1 (left).</i> My block world.
</td>
<td align="center" bgcolor="white">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_simplevisionsystem/outputs/blockworld_downloaded.jpeg" alt="Block world from chapter 2." width="300"/><br>
<i>Figure 1 (right).</i> Block world from chapter 2.
</td>
</tr>
</table>

I next identified edges and the associated gradient orientation and magnitude at each edge (**Figure 2, top left**, **Figure 2, top right**), classified edges as horizontal or vertical, and estimated the figure-ground segmentation based on color/saturation (**Figure 2, bottom left**, **Figure 2, bottom right**).

<table align="center">
<tr>
<td align="center" bgcolor="white">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_simplevisionsystem/outputs/edges.jpeg" alt="Map of edges detected with image gradient." width="300"/><br>
<i>Figure 2 (top left).</i> Map of edges detected with image gradient.
</td>
<td align="center" bgcolor="white">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_simplevisionsystem/outputs/gradmag.jpeg" alt="Map of gradient orientation and magnitude along edges." width="300"/><br>
<i>Figure 2 (top right).</i> Map of gradient orientation and magnitude along edges.
</td>
</tr>
<tr>
<td align="center" bgcolor="white">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_simplevisionsystem/outputs/figureground.jpeg" alt="Figure-ground relationships." width="300"/><br>
<i>Figure 2 (bottom left).</i> Estimated figure-ground relationships.
</td>
<td align="center" bgcolor="white">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_simplevisionsystem/outputs/edgesclassified.jpeg" alt="Classified edges." width="300"/><br>
<i>Figure 2 (bottom right).</i> Classified edges.
</td>
</tr>
</table>

I then segmented faces (**Figure 3, left**) and labeled each face as 'horizontal' or 'vertical' (**Figure 3, right**).

<table align="center">
<tr>
<td align="center" bgcolor="white">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_simplevisionsystem/outputs/faces.jpeg" alt="Detected planar faces." width="300"/><br>
<i>Figure 3 (left).</i> Detected planar faces.
</td>
<td align="center" bgcolor="white">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_simplevisionsystem/outputs/labeledfaces.jpeg" alt="Labeled planar faces." width="300"/><br>
<i>Figure 3 (right).</i> Labeled planar faces.
</td>
</tr>
</table>


I then constructed the constraint matrix based on the seven edge, face, and figure-ground constraints enumerated in the comments in the code. The recovered 3D structure is rendered from different viewpoints below(**Figure 4, bottom**) and visualized in heat maps (**Figure 4, bottom**).

<table align="center">
<tr>
<td align="center" bgcolor="white">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_simplevisionsystem/outputs/render3d.jpeg" alt="Rendered 3D reconstruction." width="450"/><br>
<i>Figure 4 (top).</i> Rendered 3D reconstruction.
</td>
</tr>
<tr>
<td align="center" bgcolor="white">
<img src="https://github.com/et22/paper-implementations/blob/main/torralba2024_simplevisionsystem/outputs/worldcoords.jpeg" alt="Recovered 3D structure in world coordinates." width="450"/><br>
<i>Figure 4 (bottom).</i> Recovered 3D structure in heat maps.
</td>
</tr>

</table>


### References
Wrote the feature extraction code from scratch but used assistance from Claude to write and solve the constraint matrix, format the README, and generate plots. 