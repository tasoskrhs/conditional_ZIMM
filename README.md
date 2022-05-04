# GAN-based Training of Semi-Interpretable Generators for Biological Data Interpolation and Augmentation
## authors: Anastasios Tsourtis, Georgios Papoutsoglou, Yannis Pantazis

### Paper: [website_name.com](https://uoc.gr)
This repository hosts the official TensorFlow implementation of our paper, 
accepted at MDPI.

## Environment Setup
After cloning the repository run (while sourcing it) `setup.sh`, this will create and activate a python 3.6 environment called **tf2_env_py36** using conda 
and will install all required packages.

`. ./setup.sh`

## Preparation
* Download the data from:

## Train

``` python
python main_example_name.py --help
usage: main_example_name.py [-h] [--data_fname] [--steps] [--d] [--mb] [--beta] [--gamma] 
                             [--K] [--K_lip] [--lam_gp] [--Z_dim] [--y_dim] [--spen]
                             [--lr]  [--saved_model]


optional arguments:
  -h, --help            show this help message and exit
  --data_fname          directory of input data
  --steps               number of training steps      
  --d                   number of dimensions of input data
  --mb                  mini-batch size (default: 1024)
  --alpha               alpha parameter of cumulant loss (default: 0.5)
  --K                   number of GMM modes (default: 30)
  --K_lip               Lipschitz constant K (default: 1.0)
  --lam_gp              gradient penalty coefficient (default: 1.0)
  --Z_dim               noise dimension of generator
  --y_dim               dimension of label embedding (default: 10)
  --spen                use Sigma Penalty on generator (default: 0.001) 
  --lr                  learning rate (default: 0.0002)
  --saved_model         name of the saved model checkpoints
  --output_fname        name of the output file directory, for this experiment

```

- ### Example 1:Train Lipschitz GAN using 2-D SwissRoll input data:

Training has two options for the generator network:
1. Train the network over a Swiss-Roll using conditional feedforward neural networks (cFNN):
``` python
python main_swiss_LIP.py  --steps=10000
```
2. Train the network over a Swiss-Roll using conditional Gaussian Mixture Model (cGMM):
```
python main_swiss_LIP_GMM.py 
```

* ### Example 2: Synthetic mRNA-seq data
1. Train the network over a Swiss-Roll using conditional feedforward neural networks (cFNN):
``` python
python main_synth_data_LIP.py --steps=100000
```
2. Train the network over a Swiss-Roll using conditional zero-inflated Mixture Model (cZIMM):
``` python
python main_synth_data_LIP_ZIMM.py --steps=100000
```

* ### Example 3: Real mRNA-seq data


## Inference