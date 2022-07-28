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

## Results
We test our method in WGAN and CE-GZSL, and here are the results.
<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="3">AWA2</th>
    <th colspan="3">CUB</th>
    <th colspan="3">SUN</th>
    <th colspan="3">APY</th>
  </tr>
  <tr>
    <th>A<sup>S</sup></th>
    <th>A<sup>U</sup></th>
    <th>A<sup>H</sup></th>
    <th>A<sup>S</sup></th>
    <th>A<sup>U</sup></th>
    <th>A<sup>H</sup></th>
    <th>A<sup>S</sup></th>
    <th>A<sup>U</sup></th>
    <th>A<sup>H</sup></th>
    <th>A<sup>S</sup></th>
    <th>A<sup>U</sup></th>
    <th>A<sup>H</sup></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Xian_Feature_Generating_Networks_CVPR_2018_paper.html">f-CLSWGAN</a></td>
    <td>57.7</td>
    <td>71.0</td>
    <td>63.7</td>
    <td>59.4</td>
    <td>63.3</td>
    <td>61.3</td>
    <td>46.2</td>
    <td>35.2</td>
    <td>40.0</td>
    <td>32.5</td>
    <td>57.2</td>
    <td>41.5</td>
  </tr>
  <tr>
    <td><b>ZLAP</b>WGAN</td>
    <td><b>65.4</td>
    <td><b>82.2</td>
    <td><b>72.8</td>
    <td><b>73.0</td>
    <td>64.8</td>
    <td><b>68.7</td>
    <td>50.1</td>
    <td><b>38.0</td>
    <td><b>43.2</td>
    <td><b>40.2</td>
    <td>53.8</td>
    <td>46.0</td>
  </tr>
  <tr>
    <td><a href="https://openaccess.thecvf.com/content/CVPR2021/html/Han_Contrastive_Embedding_for_Generalized_Zero-Shot_Learning_CVPR_2021_paper.html">CE-GZSL</a></td>
    <td>65.3</td>
    <td>75.0</td>
    <td>69.9</td>
    <td>66.9</td>
    <td>65.9</td>
    <td>66.4</td>
    <td><b>52.4</td>
    <td>34.3</td>
    <td>41.5</td>
    <td>28.3</td>
    <td><b>65.8</td>
    <td>39.6</td>
  </tr>
  <tr>
    <td><b>ZLAP</b>+CE-GZSL</td>
    <td>64.8</td>
    <td>80.9</td>
    <td>72.0</td>
    <td>71.2</td>
    <td><b>66.2</td>
    <td>68.6</td>
    <td>50.9</td>
    <td>35.7</td>
    <td>42.0</td>
    <td>38.3</td>
    <td>60.9</td>
    <td><b>47.0</td>
  </tr>
</tbody>
</table>

## Citation
If you recognize our work, please cite:  

    @inproceedings{ijcai2022-114,
        title     = {Zero-Shot Logit Adjustment},
        author    = {Chen, Dubing and Shen, Yuming and Zhang, Haofeng and Torr, Philip H.S.},
        booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, {IJCAI-22}},
        publisher = {International Joint Conferences on Artificial Intelligence Organization},
        editor    = {Lud De Raedt},
        pages     = {813--819},
        year      = {2022},
        month     = {7},
        note      = {Main Track}
        doi       = {10.24963/ijcai.2022/114},
        url       = {https://doi.org/10.24963/ijcai.2022/114},
        }
    
## Acknowledgment
We acknowledge the prior works [f-CLSWGAN](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning) and [CE-GZSL](https://github.com/Hanzy1996/CE-GZSL) for their contributions to our work.

