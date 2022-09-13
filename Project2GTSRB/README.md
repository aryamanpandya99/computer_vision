**Project 2: Transfer Learning on the German Traffic Sign Recognition Benchmark (GTSRB)**

GTSRB provides a dataset with 50,000 images of traffic signs in Germany. There exist 43 classes within the dataset, 
and it proves to be an interesting Computer Vision challenge due to the visual similarity of many of these signs. 

The goal of this project was to (as promised in Project 1) make my PyTorch code a little bit more modular, and to 
apply the concept of Transfer Learning in PyTorch. In this implementation I download/load the dataset, transform all the images 
to be somewhat normalized (and sized as equally as possible), finetune (and re-train) the original ResNet architecture to be leveraged 
for the GTSRB classification task and tested this implementation. 

On the initial iteration, I was able to obtain 90% validation accuracy and 68% test accuracy (big sad) but this makes sense because the data
was not pre-processed well enough. This is fine because optimal data pre-processing was outside the scope of this project. Future projects 
(Project 3 onwards) will have a heavier emphasis on pre-processing, further architecture fine-tuning and test set accuracy. 
