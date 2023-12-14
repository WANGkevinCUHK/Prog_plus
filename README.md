<h1 align='left'>
ProG Plus (Updating)
</h1>

|  | pre-training task |  |  | prompt design |  |  | downstream tasks |  |  | \| answering function |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Paper | node | edge | graph | prompt components | inserting pattern | prompt tuning | I node | edge | graph | Preset | Learnable |
| GPPT <br> $(\mathrm{KDD} 2022[78])$ | $x$ | $\checkmark$ | $x$ | structure token: <br> $\mathbf{s}_v \in \mathbb{R}^d$ <br> task token: <br> $\mathbf{c}_y \in \mathbb{R}^d$ | $\begin{aligned} $\mathbf{s}_{v_i} & \leftarrow f_\theta\left(v_i\right) \\\tilde{\mathbf{s}}_{y, v_i} & \leftarrow\left[\mathbf{c}_y, \mathbf{s}_{v_i}\right]\end{aligned}$$ | Cross Entropy | $\checkmark$ | $x$ | $x$ | $\checkmark$ | $x$ |
| GPF <br> (arXiv [11]) | $\checkmark$ | $\checkmark$ | $\checkmark$ | prompt feature $\mathbf{p} \in \mathbb{R}^d$ | $\tilde{\mathbf{s}}_i \leftarrow \mathrm{x}_i+\mathrm{p}$ | $\max _{p, \phi} \sum_{\left(y_i, \tilde{s}_i\right)}$ <br> $p_{\pi, \phi}\left(y_i \mid \mathbf{s}_i\right)$ | $x$ | $x$ | $\checkmark$ | $x$ | $\checkmark$ |
| All in One <br> (KDD 2023 [80]) | $x$ | $x$ | $\checkmark$ | prompt token: <br> $\mathcal{P}=\left\{\mathbf{p}_1, \ldots, \mathbf{p}_{\|\mathcal{P}\|}\right\}$ <br> token structure: <br> $\left\{\left(\mathbf{p}_i, \mathbf{p}_j\right) \mid \mathbf{p}_i, \mathbf{p}_j \in \mathcal{P}\right\}$ | $w_{i k} \leftarrow \sigma\left(\mathbf{p}_k \cdot \mathbf{x}_i^T\right)$ <br> if $\sigma\left(\mathbf{p}_k \cdot \mathbf{x}_i^T\right)>\delta$ else 0 <br> $\tilde{\mathbf{s}}_i \leftarrow \mathbf{x}_i+\sum_{k=1}^{\|\mathcal{P}\|} w_{i k} \mathbf{p}_k$ | Meta-Learning | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| GraphPrompt <br> (WWW 2023[52]) | $x$ | $\checkmark$ | $x$ | prompt token: <br> $\mathbf{p}_t \in \mathbb{R}^d, t \in \mathcal{T}$ <br> structure token: $\mathbf{s} \in \mathbb{R}^d$ <br> task token: $\mathbf{c}_y \in \mathbb{R}^d$ | $\tilde{\mathbf{s}}_i^t \leftarrow \operatorname{Readout}\left(\left\{\mathbf{p}_t \odot f_\pi(v) \mid\right.\right.$ <br> $\left.\left.v \in V\left(\mathcal{S}_i\right)\right\}\right)$ <br> $\mathbf{c}_y \leftarrow \operatorname{Mean}\left(\left\{\tilde{\mathbf{s}}_j^t \mid y_j=y\right\}\right)$ | $\min _{\mathrm{p}_t}-\sum_{\left(y_i, \mathcal{S}_i\right)} \ln$ <br> $\frac{\exp \left(\operatorname{sim}\left(s_i^t, c_{y_i}\right) / \tau\right)}{\sum_{y \in Y} \exp \left(\operatorname{sim}\left(\tilde{s}_i^t, c_y\right) / \tau\right)}$ | $\checkmark$ | $x$ | $\checkmark$ | $\checkmark$ | $x$ |

<h5 align="left">

![](https://img.shields.io/badge/Latest_version-v0.1.5-red)
![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![Testing Status](https://img.shields.io/badge/PyTorch-v1.13.1-red)
![Testing Status](https://img.shields.io/badge/license-MIT-blue)
![Testing Status](https://img.shields.io/badge/python->=3.9-red)

</h5>

<br>

🌟 ``ProG Plus`` is a baby of **ProG++**, an extended library upon [![](https://img.shields.io/badge/ProG-red)](https://github.com/sheldonresearch/ProG). ``ProG Plus`` supports more graph prompt models, and we will merge ``ProG Plus`` to [![](https://img.shields.io/badge/ProG-red)](https://github.com/sheldonresearch/ProG) in the near future (named as **ProG++**). Some implemented models are as follows (_We are now implementing more related models and we will keep integrating more models to ProG++_):  
>- ( _**KDD23 Best Paper**_ 🌟)  X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan, “All in One: Multi-Task Prompting for Graph Neural Networks,” in KDD, 2023
>- M. Sun, K. Zhou, X. He, Y. Wang, and X. Wang, “GPPT: Graph Pre-Training and Prompt Tuning to Generalize Graph Neural Networks,” in KDD, 2022
>- T. Fang, Y. Zhang, Y. Yang, and C. Wang, “Prompt tuning for graph neural networks,” arXiv preprint, 2022.
>- T. Fang, Y. Zhang, Y. Yang, C. Wang, and L. Chen, “Universal Prompt Tuning for Graph Neural Networks,” in NeurIPS, 2023.


<h5 align='center'>
  
Thanks to Dr. Xiangguo Sun for his

[![](https://img.shields.io/badge/Python_Library-ProG-red)](https://github.com/sheldonresearch/ProG)

Please visit their [website](https://github.com/sheldonresearch/ProG) to inquire more details on **ProG**, **ProG Plus**, and **ProG++**

</h5>


## Package Dependencies
--cuda 11.8

--python 3.9.17 

--pytorch 2.0.1 

--torch-geometric 2.3.1
[[quick start](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html )]


``pip install torch_geometric``

``pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html # Optional dependencies``

or 
``conda install pyg -c pyg``

## Usage

See in [https://github.com/sheldonresearch/ProG](https://github.com/sheldonresearch/ProG)

## Citation

bibtex

```
@inproceedings{sun2023all,
  title={All in One: Multi-Task Prompting for Graph Neural Networks},
  author={Sun, Xiangguo and Cheng, Hong and Li, Jia and Liu, Bo and Guan, Jihong},
  booktitle={Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery \& data mining (KDD'23)},
  year={2023},
  pages = {2120–2131},
  location = {Long Beach, CA, USA},
  isbn = {9798400701030},
  url = {https://doi.org/10.1145/3580305.3599256},
  doi = {10.1145/3580305.3599256}
}

```
