<h1 align='left'>
ProG Plus (Updating)
</h1>


<h5 align="left">

![](https://img.shields.io/badge/Latest_version-v0.1.5-red)
![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![Testing Status](https://img.shields.io/badge/PyTorch-v2.0.1-red)
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

## TODO List

> **Note**
> <span style="color:blue"> Current experimental datasets: Cora/Citeseer/Pubmed/MUTAG</span>

- [ ] Dataset:  support graph-level datasets, PROTEINS, IMDB-BINARY, REDDIT-BINARY, ENZYMES;
- [ ] Pre_train: implementation of DGI, contextpred
- [ ] Prompt: Gprompt(WWW23)
- [ ] induced graph
      
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
