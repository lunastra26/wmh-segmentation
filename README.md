## A Stacked Generalization of 3D Orthogonal Deep Learning Convolutional Neural Networks for Improved Detection of White Matter Hyperintensities in 3D FLAIR Images

### Introduction
Accurate and reliable detection of white matter hyperintensities (WMH) and its volume quantification can provide valuable clinical information to assess neurological disease progression. 
In this work, a stacked generalization ensemble of orthogonal 3D Convolutional Neural Networks (CNNs), StackGen-Net, is explored for improving automated detection of WMH in 3D T2-FLAIR images.

### Architecture
Figure shows the stacked generalization ensemble framework (StackGen-Net) used in this work. 
Individual CNNs (DeepUNET3D) in StackGen-Net were trained on 2.5D patches from orthogonal reformatting of 3D-FLAIR volumes in the training set to yield WMH posteriors. 
A Meta-CNN was trained to learn the functional mapping from orthogonal WMH posteriors to the final WMH prediction. 
Additional model architecture and training details are available in the manuscript referenced in the Citation section.

<img src="https://github.com/lunastra26/wmh-segmentation/blob/main/Images/Architecture.jpg" width="400">

### Predictions
Following assumptions are made regarding the test FLAIR volume:
1) Volumes are 3D (approximately isotropic resolution to facilitate reformatting without interpolation).
2) Volumes are in nifti format.
3) Volumes are pre-processed (brain extraction and N4 bias correction)
4) Volumes are oriented axially.

Figure shows axial and sagittal views of WMH mask predicted by StackGen-Net on a test FLAIR volume. Manual annotations are shown for reference.

<img src="https://github.com/lunastra26/wmh-segmentation/blob/main/Images/Predictions.jpg" width="400">

### Description
The following are available in this repository
1) Pretrained Orthogonal CNN models for predicting WMHs on orthogonal reformatting of 3D FLAIR volumes
2) An evaluation script for testing new 3D FLAIR volume
3) Utilities 


### Environment
Tensorflow 1.4
python 3.5
keras 2.2
nibabel 2.3

### Citation
If you use this CNN model in your work, please cite the following:
Umapathy, L., G. G. Perez-Carrillo, M. B. Keerthivasan, J. A. Rosado-Toro, M. I. Altbach, B. Winegar, C. Weinkauf, A. Bilgin, and Alzheimer’s Disease Neuroimaging Initiative. "A Stacked Generalization of 3D Orthogonal Deep Learning Convolutional Neural Networks for Improved Detection of White Matter Hyperintensities in 3D FLAIR Images." American Journal of Neuroradiology, 2021 (http://doi.org/10.3174/ajnr.A6970).

### Remarks: RESEARCH USE ONLY, NOT APPROVED FOR CLINICAL USE
