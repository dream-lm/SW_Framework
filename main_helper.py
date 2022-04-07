'''
Author: Zhou Hao
Date: 2022-04-07 17:43:11
LastEditors: Zhou Hao
LastEditTime: 2022-04-07 17:53:10
Description: file content
E-mail: 2294776770@qq.com
'''
import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_circles, make_moons
from sklearn.cluster import KMeans
from collections import Counter

def add_flip_noise(dataset:np.ndarray, noise_rate:float) -> np.ndarray:
    '''
    return : dataset with noise
    '''

    label_cat = list(set(dataset[:, 0]))
    new_data = np.array([])
    flag = 0
    for i in range(len(label_cat)):
        label = label_cat[i]
        other_label = list(filter(lambda x: x != label, label_cat))
        data = dataset[dataset[:, 0] == label]
        n = data.shape[0]
        noise_num = int(n * noise_rate)
        noise_index_list = []  # recode the index of noise
        n_index = 0
        while True:
            rand_index = int(random.uniform(0, n))
            if rand_index in noise_index_list:
                continue
            if n_index < noise_num:
                data[rand_index, 0] = random.choice(other_label)  
                n_index += 1
                noise_index_list.append(rand_index)
            if n_index >= noise_num:
                break
        if flag == 0:
            new_data = data
            flag = 1
        else:
            new_data = np.vstack([new_data, data])
    return new_data



def load_data(data_name, random_state=0):
    np.random.seed(random_state)
    if data_name == 'make_moons':
        x, y = make_moons(n_samples=1200, noise=0.4)
        data = np.hstack((y.reshape((len(y), 1)), x))
        np.random.shuffle(data)

        data_0 = data[data[:, 0] == 0]
        data_1 = data[data[:, 0] == 1]

        data_0 = data_0[:100]

        data = np.vstack((data_0, data_1))

    elif data_name == 'make_circles':
        x, y = make_circles(n_samples=1200, noise=0.3, factor=0.6)
        data = np.hstack((y.reshape((len(y), 1)), x))
        np.random.shuffle(data)
        print(Counter(data[:, 0]))

        data_0 = data[data[:, 0] == 0]
        data_1 = data[data[:, 0] == 1]
        data_0 = data_0[:100]
        data = np.vstack((data_0, data_1))

    data = pd.DataFrame(data)
    y = data[0]
    X = data[[1, 2]]
    X = X.values
    y = y.values
    return X, y