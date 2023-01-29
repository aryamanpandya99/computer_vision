**Project 3: Semantic Segmentation** 

For this project I attempted to build a Fully Convolutional Semantic Segmentation Network inspired by the Original FCN paper (Darrell et. al 2015). 

The dataset I used is the CityScapes dataset (https://www.cityscapes-dataset.com/) that provides us with thousands of images of urban street scenes, as 
well as semantic masks that classify every pixel within the image. The dataset provides masks with boath coarse and fine labeling, but for the purpose 
of this project I only utilized the fine masks to produce a smoother output. 

The workflow (goals) of this project were to first solve the segmentation task on the dataset using a FCN32, which doesn't use the skip architecture detailed 
in the paper. Then, once all the kinks were ironed out, to build a second model that integrates this skip-architecture. ,<-- this hasn't been done yet. 

The model I built was pretrained on ImageNet like those in the paper. Specifically, I chose the VGG16 pretrained model because it was evaluated 
as the best option in the paper. 

This project was more complex than the previous one, and debugging (even though issues were always found to be minor) helped me really understand the 
architecture for SegNets through and through. 
