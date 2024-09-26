**Project 2: Transfer Learning on the German Traffic Sign Recognition Benchmark (GTSRB)**

**(In-progress)**

GTSRB provides a dataset with 50,000 images of traffic signs in Germany. There exist 43 classes within the dataset, 
and it proves to be an interesting Computer Vision challenge due to the visual similarity of many of these signs. 

The goal of this project was to (as promised in Project 1) make my code a little bit more modular, and to 
apply the concept of Transfer Learning in PyTorch. In this implementation I download/load the dataset, transform all the images 
to be somewhat normalized (and resized equally), finetune (and re-train) the original ResNet architecture to be leveraged 
for the GTSRB classification task and tested this implementation. 

The Project2GTSRB.ipynb notebook contains the code for the model and classification implementation, whereas the GTSRBDataAugmentation.ipynb notebook contains the code for data visualization and augmentation. While the model may seem like the attractive and cool part, the data augmentation and visualization provide the strategy to make the model accurate. A lot more time and effort went into that side of things. This applies for this dataset since its unbalanced in its class distribution, but is a trend you can observe in most real world data. We tend to see some sort of Gaussan distribution of most data. 
