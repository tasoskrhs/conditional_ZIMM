# GAN-based Training of Semi-Interpretable Generators for Biological Data Interpolation and Augmentation
## authors: Anastasios Tsourtis, Georgios Papoutsoglou, Yannis Pantazis

### Paper: [https://doi.org/10.3390/app12115434](https://www.mdpi.com/2076-3417/12/11/5434/htm)
This repository hosts the official TensorFlow implementation of our paper, 
published at MDPI.

## Environment Setup
We assume that conda is already installed on your system.
After cloning the repository run `setup.sh` script (while sourcing it): 

`. ./setup.sh`

It will create and activate a python 3.9 environment called **tf2_env_py39** using conda 
and will install all required packages.


Please refer to https://www.tensorflow.org/install/pip#linux_1 for more information on how to set up
TensorFlow with GPU acceleration. GPU acceleration is preferable but not mandatory.

## Input data
Please find the data corresponding to each experiment under `\input_data` . 
For further information refer to the published paper and `data_loading.py`


## Training

``` python
python main_example_name.py --help
usage: main_example_name.py [-h] [--data_fname] [--steps] [--d] [--mb] [--beta] [--gamma] 
                             [--K] [--K_lip] [--lam_gp] [--Z_dim] [--y_dim] [--spen]
                             [--lr]  [--saved_model] [--output_fname] [--resume_from_iter]
                             [--missing_labels] [--generate]


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
  --resume_from_iter    steps corresponding to last checkpoint, needed to resume training
  --missing_labels      Missing labeled data in the training set. Options: 'none' (default), '0.4_0.6', 'state_2'
  --generate            Generate samples, provided training has taken place (default: False)
```

- ### Example 1: Train Lipschitz GAN using a 2-D SwissRoll dataset

We provide two options for the generator network:
1. Train using the conditional feedforward neural network (cFNN) generator for 100K steps:
``` python
python main_swiss_LIP.py  --steps 100000
```
2. Train using the conditional Gaussian Mixture Model (cGMM) generator and sigma penalty coefficient to be 0.002:
```
python main_swiss_LIP_GMM.py -spen 0.002
```
The experiment parameters are saved in `\output_files\experiment_name\commandline_args.txt`.
* ### Example 2: Synthetic RNA-seq data
1. Train using the conditional feedforward neural network (cFNN) generator with gating on the output layer. Training data labels lie uniformly in [0, 1]:
``` python
python main_synth_data_LIP.py --steps 200000 --missing_labels none
```
2. Train using the conditional zero-inflated Mixture Model (cZIMM) generator, where interval [0.4:0.6] is missing from the training data set:
``` python
python main_synth_data_LIP_ZIMM.py --missing_labels 0.4_0.6
```

* ### Example 3: Real single-cell mass cytometry data
1. Train using the conditional feedforward neural network (cFNN) generator with gating on the output layer. 
   Here we continue training from a previous run of 100K steps for another 200K steps:
``` python
python main_real_data_LIP.py --steps 300000 --resume_from_iter 100000
```
2. Train using the conditional zero-inflated Mixture Model (cZIMM) generator:
``` python
python main_real_data_LIP_ZIMM.py 
```

## Inference
During training, generated data, plots and running losses are being exported in the corresponding subdirectories under `\output_files\experiment_name\`.
Assuming training has already taken place (checkpoint files containing model weights have been created), we can further generate data by:
``` python
python main_example_name.py --generate
```
The generated data are:
* .csv file containing generated points (rows: samples, columns: dimensions)
* corresponding plots (similar to the ones during training)

Further training can be carried out as shown in the previous section. 

<!-- 
## Visualization
PCA plots...
-->

 
## Questions
Please don't hesitate to send me an email.
