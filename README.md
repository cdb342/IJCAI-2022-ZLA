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

## Result
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-1wig{font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">Method</th>
    <th class="tg-c3ow" colspan="3">AWA2</th>
    <th class="tg-baqh" colspan="3">CUB</th>
    <th class="tg-baqh" colspan="3">SUN</th>
    <th class="tg-baqh" colspan="3">APY</th>
  </tr>
  <tr>
    <th class="tg-c3ow">A^U^</th>
    <th class="tg-c3ow">A^S^</th>
    <th class="tg-c3ow">A^H^</th>
    <th class="tg-baqh">A^U^</th>
    <th class="tg-baqh">A^S^</th>
    <th class="tg-baqh">A^H^</th>
    <th class="tg-baqh">A^U^</th>
    <th class="tg-baqh">A^S^</th>
    <th class="tg-baqh">A^H^</th>
    <th class="tg-baqh">A^U^</th>
    <th class="tg-baqh">A^S^</th>
    <th class="tg-baqh">A^H^</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">f-CLSWGAN</td>
    <td class="tg-0lax">57.7</td>
    <td class="tg-0lax">71.0</td>
    <td class="tg-0lax">63.7</td>
    <td class="tg-0lax">59.4</td>
    <td class="tg-0lax">63.3</td>
    <td class="tg-0lax">61.3</td>
    <td class="tg-0lax">46.2</td>
    <td class="tg-0lax">35.2</td>
    <td class="tg-0lax">40.0</td>
    <td class="tg-0lax">32.5</td>
    <td class="tg-0lax">57.2</td>
    <td class="tg-0lax">41.5</td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:bold">ZLAP</span>WGAN</td>
    <td class="tg-1wig">65.4</td>
    <td class="tg-1wig">82.2</td>
    <td class="tg-1wig">72.8</td>
    <td class="tg-1wig">73.0</td>
    <td class="tg-0lax">64.8</td>
    <td class="tg-1wig">68.7</td>
    <td class="tg-0lax">50.1</td>
    <td class="tg-1wig">38.0</td>
    <td class="tg-1wig">43.2</td>
    <td class="tg-1wig">40.2</td>
    <td class="tg-0lax">53.8</td>
    <td class="tg-0lax">46.0</td>
  </tr>
  <tr>
    <td class="tg-0lax">CE-GZSL</td>
    <td class="tg-0lax">65.3</td>
    <td class="tg-0lax">75.0</td>
    <td class="tg-0lax">69.9</td>
    <td class="tg-0lax">66.9</td>
    <td class="tg-0lax">65.9</td>
    <td class="tg-0lax">66.4</td>
    <td class="tg-1wig">52.4</td>
    <td class="tg-0lax">34.3</td>
    <td class="tg-0lax">41.5</td>
    <td class="tg-0lax">28.3</td>
    <td class="tg-1wig">65.8</td>
    <td class="tg-0lax">39.6</td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:bold">ZLAP</span>+CE-GZSL</td>
    <td class="tg-0lax">64.8</td>
    <td class="tg-0lax">80.9</td>
    <td class="tg-0lax">72.0</td>
    <td class="tg-0lax">71.2</td>
    <td class="tg-1wig">66.2</td>
    <td class="tg-0lax">68.6</td>
    <td class="tg-0lax">50.9</td>
    <td class="tg-0lax">35.7</td>
    <td class="tg-0lax">42.0</td>
    <td class="tg-0lax">38.3</td>
    <td class="tg-0lax">60.9</td>
    <td class="tg-1wig">47.0</td>
  </tr>
</tbody>
</table>
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

