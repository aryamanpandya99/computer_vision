# Computer Vision

In this repository, I implement computer vision techniques and ideas that are initially new to me in notebooks. 
Then, if the notebook contains blocks of code that can be re-used for different vision tasks, I put them in files 
under the `src` directory. 

In this iterative fashion, I'm building a mini vision library built off of pure PyTorch. 

This repository goes from Lenet-5, to diffusion. 

## Notebooks

- `mnist_lenet.ipynb`: training LeNet-5 on MNIST. 
- `autoencoder.ipynb`: autoencoders. 
- `dcgan.ipynb`: Deep convolutional Generative Adversarial Networks. 
- `resnets.ipynb`: residual networks.  
- `fcn_Segmentation.ipynb`: fully convolutional networks for semantic segmentation. 
- `ddpm_mnist.ipynb`: implementing diffusion models architecture on MNIST.
- `ddpm_cifar.ipynb`: applying the DDPM architecture from `src` to CIFAR-10.
- `mixture_of_experts.ipynb`: implementing a sparsely-gated mixture of experts with ResNets.
- `ddim.ipynb`: implementing denoising diffusion implicit models.
