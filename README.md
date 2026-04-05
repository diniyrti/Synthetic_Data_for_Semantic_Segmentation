## Synthetic Dataset Deep Learning
Automatic information extraction from massive satellite imagery data, especially using deep learning methods, is very challenging because it requires a large amount of accurate, error-free, and manually annotated training data. Therefore, deep learning model building and evaluation are performed using synthetic datasets to reduce the burden of time-consuming and expensive manual annotation. The creation of this synthetic dataset refers to the research of Clabaut et al. 2024.

This research's synthetic dataset was constructed using the Object-Based Image Analysis (OBIA) approach. The process begins with segmenting Sentinel-2 imagery using the Large Scale Mean Shift (LSMS) algorithm in QGIS. LSMS is capable of identifying homogeneous objects in satellite imagery while maintaining the geometric and statistical integrity of each segment (OTB Development Team, 2024), thereby producing homogeneous/pure land use class objects for subsequent processing. Figure 1 displays the Large Scale Mean Shift (LSMS) algorithm settings interface in QGIS software for the satellite image segmentation process. Spatial parameters and range radius are configured to determine the level of detail, with lower values resulting in more detailed polygon segments. This stage serves to convert raster data into vector polygon analysis units.

<p align="center">
  <img src="https://github.com/diniyrti/Synthetic_Data_for_Semantic_Segmentation/blob/main/images/Figure%201.jpg?raw=true" width="800">
</p>

The polygons then serve as basic units of analysis that are manually labeled with land use classes by overlapping the segmentation vectors with Planet Scope imagery and Sentinel-2 false color composites to facilitate interpretation, as seen in Figure 2. The labeled vector polygons are then clipped with Sentinel 2 raster data to produce a collection of pure (homogeneous) land use polygon fragments from satellite imagery (Sentinel 2 MSI) that are used for synthetic data generation (Figure 3). In addition, synthetic backgrounds are created as a crucial component in the training data generation process. The creation of synthetic backgrounds aims to introduce environmental variability into the dataset and prevent overfitting of the model on uniform samples. The size of the square background taken is 64x64 pixels adjusted to the size of the homogeneous background at the research location (Figure 4).

<p align="center">
  <img src="https://github.com/diniyrti/Synthetic_Data_for_Semantic_Segmentation/blob/main/images/Figure%202.jpg?raw=true" width="500">
</p>

<p align="center">
  <img src="https://github.com/diniyrti/Synthetic_Data_for_Semantic_Segmentation/blob/main/images/Figure%203.jpg?raw=true" width="500">
</p>

<p align="center">
  <img src="https://github.com/diniyrti/Synthetic_Data_for_Semantic_Segmentation/blob/main/images/Figure%204.jpg?raw=true" width="500">
</p>

Next, the mixing process is performed. The mixing process for the sample boundaries to be simulated is carried out using the following steps. An example of the synthetic data results can be seen in Figure 6:

•	Creation of a binary mask that allows for cutting the background;

•	Application of a 3x3 "average" type convolution on the binary mask to obtain mixing proportions on the object’s boundaries, called a "soft" mask hereafter;

•	Cutting the background with softening of the edges;

•	Softening of the edges for the object to be added;

•	Sum of the sample and the background mosaic weighted by the filtered mask;

•	The ground truth is the mask used to "paste" the sample.

<p align="center">
  <img src="https://github.com/diniyrti/Synthetic_Data_for_Semantic_Segmentation/blob/main/images/Figure%205.jpg?raw=true" width="500">
</p>

<p align="center">
  <img src="https://github.com/diniyrti/Synthetic_Data_for_Semantic_Segmentation/blob/main/images/Figure%206.jpg?raw=true" width="500">
</p>

## Synthetic Dataset for Land Use Classification Using UNet
This project implements a 4-level encoder-decoder U-Net architecture designed for the multiclass semantic segmentation of Sentinel-2 MSI imagery. Utilizing 10 spectral bands, the model classifies 16 distinct land use categories. A Dice Loss function was employed during training to optimize spatial overlap and maintain the integrity of complex land boundaries.

<p align="center">
  <img src="https://github.com/diniyrti/Synthetic_Data_for_Semantic_Segmentation_UNet/blob/main/images/Learning%20Curves%20Train%20vs%20Val.png?raw=true" width="900">
</p>

The learning curves demonstrate rapid and stable convergence, with both training and validation loss decreasing sharply within the first 20 epochs. Validation metrics reached a stable, showing a consistent trend with the training data. This alignment indicates that the model effectively captured the underlying spatial features from the synthetic dataset with minimal overfitting.

Upon final evaluation, the model achieved robust performance metrics, including a test loss of 0.219, a test accuracy of 0.775, and a mean Intersection over Union (mIoU) of 0.653. These results confirm that the framework provides a reliable foundation for automated land cover mapping and large-scale environmental monitoring applications.

<p align="center">
  <img src="https://github.com/diniyrti/Synthetic_Data_for_Semantic_Segmentation_UNet/blob/main/images/Land%20Use%20Classification%20Using%20UNet.png?raw=true" width="600">
</p>

### Reference 
Clabaut, É., Foucher, S., Bouroubi, Y., & Germain, M. (2024). Synthetic Data for Sentinel-2 Semantic Segmentation. Remote Sensing, 16(5), 818. https://doi.org/10.3390/rs16050818

