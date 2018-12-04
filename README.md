# Multi-task Sparse Group Lasso (MT-SGL) #

This repository contains a python implementation of the MT-SGL algorithm proposed in the paper [Modeling Alzheimer's Disease Cognitive Scores using
Multi-task Sparse Group Lasso](https://www.sciencedirect.com/science/article/abs/pii/S0895611117301076).

## Overview ##

The multi-task sparse group lasso (MT-SGL) framework estimates sparse features coupled across tasks, and can work with loss functions associated with any Generalized Linear Models. MT-SGL encourages 1) individual feature selection based on the utility of the features across all tasks and 2) task specific group selection based on the utility of the group to decouple the ROIs sharing across tasks allowing for more exibility. Moreover, the proposed MT-SGL framework can use general loss functions, including losses derived from generalized linear models (GLMs). Accelerated versions of two different approaches (proximal averaging and proximal composition) based on suitable FISTA-style application of accelerated gradient descent is derived to solve MT-SGL formulation. 

This code has been tested only in python in both Linux and Mac.

## How to run? ##

We created the file `test_mt_sgl.py` to show how to run MT-SGL code. 

MT-SGL considers three different loss functions settings derived from the GLM family: 'gaussian', where all scores are modeled with Gaussian regression; 'poisson', where all scores are modeled with Poisson regression; and '2g3p', where two scores are modeled with Gaussian and three scores with Poisson regression. Moreover, two proximal operators combination strategies are discussed, including proximal average and proximal composition. From this script one can change it as needed.

## Structure of the input data files ##

In order to run the code the input data files containing the training and test data must follow a specific format. The `input` of the function includes: two matrices, **X** denotes covariate matrix *n x p* with the number of *n* samples and *p* covariates; **Y** denotes response matrix *n x k* with *k* tasks; and group information vector with *p* covariates divided *q* disjoint groups. Note that, the number of features in each group can be different.

## How to cite it? ##

If you like it and want to cite it in your papers, you can use the following:

```
#!latex

@article{liu2018modeling,
  title={Modeling Alzheimer's disease cognitive scores using multi-task sparse group lasso},
  author={Liu, Xiaoli and Goncalves, Andr{\'e} R and Cao, Peng and Zhao, Dazhe and Banerjee, Arindam and Alzheimer's Disease Neuroimaging Initiative and others},
  journal={Computerized Medical Imaging and Graphics},
  volume={66},
  pages={100--114},
  year={2018},
  publisher={Elsevier}
}
```

## Have a question? ##

If you found any bug or have a question, don't hesitate to contact me:

[Xiaoli Liu]
email: `neuxiaoliliu -at- gmail -dot- com`

