# Zero-Shot Logit Adjustment
[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2204.11822-B31B1B.svg)](https://arxiv.org/abs/2204.11822)
[![Pytorch 1.8.1](https://img.shields.io/badge/pytorch-1.0.1-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cdb342/IJCAI-2022-ZLA)

This project contains the [pytorch](http://pytorch.org/) implemention for [*Zero-Shot Logit Adjustment*](https://arxiv.org/abs/2204.11822).

## Dependencies
- Python 3.7
- Pytorch = 1.0.1
- NumPy = 1.17.3
## Prerequisites
- Dataset: please download the [dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly), and change `--dataroot` to your local path.
- Semantic: The semantics for AWA2, SUN, and APY are available in the dataset. please download the 1024-D [CUB semantic](https://github.com/Hanzy1996/CE-GZSL) and save it to the data path.

## Train and Test
Please run the following commands to test on different datasets:

- python ./WGAN+ZLAP.py --dataset AWA2 --attSize 85 --nz 85 --syn_num 10 --ratio 1000  
- python ./WGAN+ZLAP.py --dataset SUN --attSize 102 --nz 102 --syn_num 10 --ratio 60  
- python ./WGAN+ZLAP.py --dataset APY --attSize 64 --nz 64 --syn_num 10 --ratio 300  
- python ./WGAN+ZLAP.py --dataset CUB --attSize 1024 --nz 1024 --syn_num 10 --ratio 30 --class_embedding sent

The meaning of these args is

- `--dataset`: datasets, e.g: SUN.  
- `--attSize`: size of semantic descriptors.  
- `--nz`: size of the Gaussian noise.  
- `--syn_num`: synthetic number for each unseen class.  
- `--reatio`: hyperparameter to control the seen-unseen prior (see Sec. 4.4 of the paper)
## Citation
If you recognize our work, please cite:  

    @inproceedings{chen2022zero,  
      title={Zero-Shot Logit Adjustment},  
      author={Chen, Dubing and Shen, Yuming and Zhang, Haofeng and Philip H.S. Torr},  
      year={2022},  
      organization={IJCAI}  
    }
    
## Acknowledgment
We acknowledge the prior works [CLSWGAN](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning) and [CE-GZSL](https://github.com/Hanzy1996/CE-GZSL) for their contributions to our work.

