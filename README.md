# Codes for IJCAI 2022 Paper: Zero-Shot Logit Adjustment [(arxiv)](https://arxiv.org/abs/2204.11822).
## Dependencies
- Python 3.7
- Pytorch 1.0.1
- numpy 1.17.3
- <mark style="background-color：green">Marked Txt</mark>
## Datasets
Please refer to [Xian et al. (CVPR2017)](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly) for the datasets (AWA2/CUB/SUN/APY), and save correspongding data into directory <font style="background: grey;">./data/.</font> Please refer to [Han et al. (CVPR 2021)](https://github.com/Hanzy1996/CE-GZSL) for the 1024-D CUB semantics.
## Train and Test
Please run the following commands to test on different datasets:

- $ python ./WGAN+ZLAP.py --dataset AWA2 --attSize 85 --nz 85 --syn_num 10 --ratio 1000  
- $ python ./WGAN+ZLAP.py --dataset CUB --attSize 1024 --nz 1024  --syn_num 10 --ratio 30  
- $ python ./WGAN+ZLAP.py --dataset SUN --attSize 102 --nz 102 --syn_num 10 --ratio 60  
- $ python ./WGAN+ZLAP.py --dataset APY --attSize 64 --nz 64 --syn_num 10 --ratio 300  

The meaning of these args is

- <font style="background: grey;">--dataset</font>: datasets, e.g: SUN.  
- <font style="background: grey;">--attSize</font>: size of semantic descriptors.  
- <font style="background: grey;">--nz</font>: size of the Gaussian noise.  
- <font style="background: gray;">--syn_num</font>: synthetic number for each unseen class.  
- --reatio: hyperparameter to control the seen-unseen prior (see Sec. 4.4 of the paper)
## Citation
If you recognize our work, please cite:  

@inproceedings{chen2022zero,  
  title={Zero-Shot Logit Adjustment},  
  author={Chen, Dubing and Shen, Yuming and Zhang, Haofeng and Philip H.S. Torr},  
  year={2022},  
  organization={IJCAI}  
}
