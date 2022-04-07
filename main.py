'''
Author: Zhou Hao
Date: 2021-02-22 14:05:42
LastEditors: Zhou Hao
LastEditTime: 2022-04-07 17:59:51
Description: 画图和测试的代码
            main_post : SMOTE_ENN',
                    'SMOTE_TomekLinks',
                    'SMOTE_RSB',
                    'SMOTE_IPF'
            main_pre : smote,adasyn,borderline1,kmenas-smote
E-mail: 2294776770@qq.com
'''

import all_smote_v3  # 加权
from imblearn import over_sampling  # 原始
from main_helper import load_data,add_flip_noise
from CRF import Crf_zhou
import minisom
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import _smote_variants_v3 as sv3    # Origin with SW
import _smote_variants_v1 as sv1    # Origin
import imbalanced_databases as imbd
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

font = {'family': 'Times New Roman',
            'size': 15, }  
font2 = {'family': 'Times New Roman',
'size': 20, }

def draw(X, y, ax, X_samp, title, num, main):
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],
                c='tan', marker='o', s=10, )
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1],
                c='darkcyan', marker='o', s=10, )
    X_new = pd.DataFrame(X_samp).iloc[len(X):, :]

    plt.scatter(X_new[0], X_new[1], c='red', s=10, marker='+')
    a = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    b = ['(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)']

    if main == 'pre':
        title = a[num % 10-1]+' '+title
    elif main == 'post':
        title = b[num % 10-1]+' '+title
    plt.title(title, font2)
    plt.ylim(ymax=1.4)  
    plt.grid()  


def main_post(data=1):

    all = [
        'SMOTE_ENN',
        'SMOTE_TomekLinks',
        'SMOTE_RSB',
        'SMOTE_IPF',
    ]
    if data == 2:
        X, y = load_data(data_name='make_moons')
    elif data == 3:
        X, y = load_data(data_name='make_circles')


    dataset = np.hstack([y.reshape(X.shape[0], 1), X])  # the 1st column is label
    RData, weight = Crf_zhou(dataset, 10)  
    X_rd, y_rd = RData[:, 1:], RData[:, 0]
    ntree = 10

    oversamplers_3 = sv3.get_all_oversamplers(all=all)  # SW
    oversamplers_1 = sv1.get_all_oversamplers(all=all)  # Original
    num = 240
    plt.figure(figsize=(20, 10))

    for o in zip(oversamplers_1, oversamplers_3):
        num += 1
        oversampler_1 = o[0]()  
        oversampler_3 = o[1]()  
        X_samp_1, y_samp_1 = oversampler_1.sample(X, y,)
        X_samp_3, y_samp_3 = oversampler_3.sample(X_rd, y_rd, weight, ntree)

        ax = plt.subplot(num)
        draw(main='post', num=num, X=X, y=y, X_samp=X_samp_1,
             ax=ax, title=oversampler_1.__class__.__name__)
        if num == 241 or num == 245:
            ax.set_ylabel('y',font,rotation=90)
        if num >= 245:
            ax.set_xlabel('x',font)
        num += 1
        ax = plt.subplot(num)
        draw(main='post', num=num, X=X_rd, y=y_rd, X_samp=X_samp_3,
             ax=ax, title='SW-'+str(oversampler_3.__class__.__name__))
        if num >= 245:
            ax.set_xlabel('x',font)

    plt.show()


def main_pre(data=1):
    all = [
        'SMOTE',
        'ADASYN',
    ]

    if data == 2:
        X, y = load_data(data_name='make_moons')
    elif data == 3:
        X, y = load_data(data_name='make_circles')

    dataset = np.hstack([y.reshape(X.shape[0], 1), X])  
    RData, weight = Crf_zhou(dataset, 10)  
    X_rd, y_rd = RData[:, 1:], RData[:, 0]
    ntree = 10

    oversamplers_3 = sv3.get_all_oversamplers(all=all)  
    oversamplers_1 = sv1.get_all_oversamplers(all=all)  
    num = 240
    plt.figure(figsize=(20, 10))

    for o in zip(oversamplers_1, oversamplers_3):
        oversampler_1 = o[0]()  
        oversampler_3 = o[1]()  
        X_samp_1, y_samp_1 = oversampler_1.sample(X, y)
        X_samp_2, y_samp_2 = oversampler_3.sample(X_rd, y_rd, weight, ntree)

        num += 1
        ax = plt.subplot(num)
        draw(main='pre', num=num, X=X, y=y, X_samp=X_samp_1,
             ax=ax, title=oversampler_1.__class__.__name__)
        if num == 241:
            ax.set_ylabel('y',font,rotation=90)

        num += 1
        ax = plt.subplot(num)
        draw(main='pre', num=num, X=X_rd, y=y_rd, X_samp=X_samp_2,
             ax=ax, title='SW-'+str(oversampler_3.__class__.__name__))


    '''boderline_1'''
    sm_1 = over_sampling.BorderlineSMOTE(random_state=42, kind="borderline-1")
    X_res, y_res = sm_1.fit_resample(X, y)
    num += 1
    ax = plt.subplot(num)
    draw(main='pre', num=num, X=X, y=y, X_samp=X_res,
        ax=ax, title='borderline_SMOTE1')
    ax.set_ylabel('y',font,rotation=90)
    ax.set_xlabel('x',font)

    '''crf-weight-boderline'''
    sm_zhou = all_smote_v3.BorderlineSMOTE(
        random_state=42, ntree=ntree, weight=weight)
    X_res, y_res = sm_zhou.fit_resample(X_rd, y_rd)
    num += 1
    ax = plt.subplot(num)
    draw(main='pre', num=num, X=X_rd, y=y_rd,
        X_samp=X_res, ax=ax, title='SW-borderline_SMOTE1')
    ax.set_xlabel('x',font)


    '''kmeans-smote'''
    sm_4 = over_sampling.KMeansSMOTE(random_state=42, )
    X_res, y_res = sm_4.fit_resample(X, y)
    num += 1
    ax = plt.subplot(num)
    draw(main='pre', num=num, X=X, y=y, X_samp=X_res, ax=ax, title='kmeans-SMOTE')
    ax.set_xlabel('x',font)

    '''crf-weight-kmeans-smote'''
    sm_3 = all_smote_v3.KMeansSMOTE(
        random_state=42, ntree=ntree, weight=weight)
    X_res, y_res = sm_3.fit_resample(X_rd, y_rd)
    num += 1
    ax = plt.subplot(num)
    draw(main='pre', num=num, X=X_rd, y=y_rd,
        X_samp=X_res, ax=ax, title='SW-kmeans-SMOTE')
    ax.set_xlabel('x',font)

    plt.show()



if __name__ == "__main__":
    main_post(data=3)
    # main_pre(data=2)
    print("Oversampling Over！")
