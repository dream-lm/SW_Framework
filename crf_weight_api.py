'''
Author: Zhou Hao
Date: 2021-02-20 17:11:14
LastEditors: Zhou Hao
LastEditTime: 2022-04-07 17:56:04
Description: 
            called by _smote_variants_v3.py
            add_weight() 是加权函数
E-mail: 2294776770@qq.com
'''
import warnings
from collections import OrderedDict
from functools import wraps
from inspect import signature, Parameter
from numbers import Integral, Real
import numpy as np
from sklearn.base import clone
from sklearn.neighbors._base import KNeighborsMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target
from imblearn.exceptions import raise_isinstance_error
from main_helper import add_flip_noise
from CRF import Crf_zhou
from sklearn.utils.fixes import np_version, parse_version
from scipy.sparse import issparse
from sklearn.utils import _safe_indexing
from itertools import compress
import random


def check_neighbors_object(nn_name, nn_object, additional_neighbor=0):
    """Check the objects is consistent to be a NN.
    Several methods in imblearn relies on NN. Until version 0.4, these
    objects can be passed at initialisation as an integer or a
    KNeighborsMixin. After only KNeighborsMixin will be accepted. This
    utility allows for type checking and raise if the type is wrong.
    Parameters
    ----------
    nn_name : str,
        The name associated to the object to raise an error if needed.
    nn_object : int or KNeighborsMixin,
        The object to be checked
    additional_neighbor : int, optional (default=0)
        Sometimes, some algorithm need an additional neighbors.
    Returns
    -------
    nn_object : KNeighborsMixin
        The k-NN object.
    """
    if isinstance(nn_object, Integral):
        return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
    elif isinstance(nn_object, KNeighborsMixin):
        return clone(nn_object)
    else:
        raise_isinstance_error(nn_name, [int, KNeighborsMixin], nn_object)


def in_danger_noise(
        nn_estimator, samples, target_class, y, kind="danger"
    ):
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)
        
        if kind == "danger":
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(          #这里-1的原因是模型初始化的时候+1了
                n_maj >= (nn_estimator.n_neighbors - 1) / 2,
                n_maj < nn_estimator.n_neighbors - 1,
            ),n_maj

        elif kind == "noise":
            # Samples are noise for m = m'
            return n_maj == nn_estimator.n_neighbors - 1,n_maj
        else:raise NotImplementedError


def add_weight(X,y,X_min,minority_label,
    base_indices,neighbor_indices,num_to_sample,
    ind,X_neighbor,X_base,weight,ntree):

    weight_maj = _safe_indexing(weight,np.flatnonzero(y == minority_label))       #原始crf权重
    new_n_maj = np.array([round((1-i/ntree),2) for i in weight_maj]) #计算后的权重

    X_base_weight = new_n_maj[base_indices]     #跟(种子节点， 母节点)节点权重矩阵
    X_neighbor_weight = new_n_maj[ind[base_indices,neighbor_indices]]   #紧邻点权重矩阵

    weights = []
    delete_index = []
    for n in range(int(num_to_sample)):
        if X_base_weight[n]!=0 and X_neighbor_weight[n]!=0: #如果母点和随机点权重都不是噪声点
            if X_base_weight[n]>= X_neighbor_weight[n]:
                proportion = (X_neighbor_weight[n] / (X_base_weight[n]+X_neighbor_weight[n])*round(random.uniform(0,1),len(str(num_to_sample))))#权重比例
            elif X_base_weight[n]< X_neighbor_weight[n]:
                proportion = X_neighbor_weight[n] / (X_base_weight[n]+X_neighbor_weight[n])
                proportion = proportion+(1-proportion)*(round(random.uniform(0,1),len(str(num_to_sample))))#权重比例
        weights.append(proportion)

    X_neighbor = np.delete(X_neighbor,delete_index,axis=0)
    X_base = np.delete(X_base,delete_index,axis=0)

    weights=np.array(weights).reshape(int(len(weights)),1)
    samples= X_base + np.multiply(weights, X_neighbor - X_base)
    return samples



