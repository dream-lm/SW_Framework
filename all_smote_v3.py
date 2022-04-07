'''crf-weight-版本
smote,borderline-smote1,svmsmote,kmeans-smote
'''
import math
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.exceptions import raise_isinstance_error
from imblearn.utils import check_neighbors_object
from imblearn.utils import Substitution
from imblearn.utils._docstring import _n_jobs_docstring
from imblearn.utils._docstring import _random_state_docstring
import random


#-----------------------------------------------------------------------------------
def make_samples_zhou(
        X, y_dtype, y_type, nn_data, nn_num, n_samples,
        new_n_maj, danger_and_safe=None, mother_point=None
):
    """
        Parameters
        ----------
        X :         {array-like, sparse matrix} of shape (n_samples, n_features)
                    Points from which the points will be created.
        y_dtype :   data type，The data type of the targets.
        y_type :    str or int
                    The minority target value, just so the function can return the
                    target values for the synthetic variables with correct length in
                    a clear format.
        nn_data :   ndarray of shape (n_samples_all, n_features)
                    Data set carrying all the neighbours to be used
        nn_num :    ndarray of shape (n_samples_all, k_nearest_neighbours)
                    The nearest neighbours of each sample in `nn_data`.
        n_samples : int
                    The number of samples to generate.
        step_size : float, default=1.0
                    The step size to create samples.

        Returns
        -------
        X_new :     {ndarray, sparse matrix} of shape (n_samples_new, n_features)
                    synthetically generated samples.
        y_new :     ndarray of shape (n_samples_new,)
                    Target values for synthetic samples.
    """
    X_new = generate_samples_zhou(X, nn_data, nn_num, n_samples,
                                new_n_maj, danger_and_safe,mother_point)
    y_new = np.full(len(X_new), fill_value=y_type, dtype=y_dtype)
    return X_new, y_new


def generate_samples_zhou(X, nn_data, nn_num, n_samples,
                        new_n_maj, danger_and_safe,mother_point=None):
    '''
        #遍历边界和安全点，根据近邻点索引矩阵找到对应的KNN个点，随机有放回抽取N次，两点坐标相减，按权重比算距离来插值，
        #剩下的m个点用来随机不放回抽点，计算抽到的点和近邻点的权重比,每个点只插一次
    '''
    mother_points = nn_data[mother_point]   
    n = n_samples // danger_and_safe        #每个点要插的数量
    m = n_samples % danger_and_safe         #多余的用来随机插值
    weidu = X.shape[1]
    X_new_1 = np.zeros(weidu)
    if not isinstance(nn_data, list):nn_data = nn_data.tolist()

    '''对整个近邻点索引矩阵按权重从小到大排序'''
    nn_num_ordered = []
    for i in range(len(nn_num)):
        values = np.array(new_n_maj)[nn_num[i]]
        keys = nn_num[i]
        dicts,num = {},[]
        for j in range(len(values)):dicts[keys[j]] = values[j]
        d_order=sorted(dicts.items(),key=lambda x:x[1],reverse=False)   #字典排序法
        for d in d_order:num.append(d[0])
        nn_num_ordered.append(num)
    length_of_num = len(num)        #近邻点矩阵的长度
    # if not isinstance(nn_data, list):nn_data = nn_data.tolist()

    '''每个点需要插的n个点'''
    for nn in range(danger_and_safe):  # 每个点的横纵坐标值
        #step_1: 获取母节点(种子节点)信息
        num = nn_num_ordered[nn]    #每个点的近邻点索引矩阵
        nn_point = mother_points[nn]      #母节点的横纵坐标
        nn_weight = new_n_maj[mother_point[nn]]      #当前母节点的权重

        for i in range(n):  # 随机有放回的抽取N次近邻点
            #step_2:    获取近邻点信息
            random_point = num[i%length_of_num]           #按权重从小到大抽取近邻点索引
            random_point_weight = new_n_maj[random_point]  # 随机点权重
            random_point_data = nn_data[random_point]  # 随机点的横纵坐标

            #step_3:    根据情况开始插值
            if nn_weight != 0 and random_point_weight != 0:  # 两个都不是噪声点
                proportion = (random_point_weight / (nn_weight + random_point_weight))  # 权重比例
                if nn_weight >= random_point_weight:
                    X_new_zhou = np.array(nn_point) + (
                                np.array(random_point_data) - np.array(nn_point)) * proportion * round(
                        random.uniform(0, 1), len(str(n_samples)))
                elif nn_weight < random_point_weight:
                    X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (
                                1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))
                X_new_1 = np.vstack((X_new_zhou, X_new_1))

    '''随机不放回抽取m个点'''
    for mm in range(m):
        #step_1:获取母节点信息
        nn_point_index = random.choice(mother_point)        #母点的索引
        nn_point_weight = new_n_maj[nn_point_index]         #母点的权重
        nn_point = nn_data[nn_point_index]                  #母点的横纵坐标
        a = np.where(mother_point == nn_point_index)[0][0]  #当前母节点在mother_point中的索引

        #step_2:获取近邻点信息
        num = nn_num[a].tolist()       #抽取点的近邻点列表
        random_point = num[0]  # 随机近邻点的索引
        random_point_weight = new_n_maj[random_point]  # 随机近邻点的权重
        random_point_data = nn_data[random_point]

        #step_3:开始插值
        if nn_point_weight != 0 and random_point_weight != 0:  # 两个都不是噪声点
            proportion = (random_point_weight / (nn_point_weight + random_point_weight))  # 权重比例
            if nn_point_weight >= random_point_weight:
                X_new_zhou = np.array(nn_point) + (np.array(random_point_data) - np.array(nn_point)) * proportion * round(random.uniform(0, 1),len(str(n_samples)))
            elif nn_point_weight < random_point_weight:
                X_new_zhou = np.array(random_point_data) + (np.array(nn_point) - np.array(random_point_data)) * (1 - proportion) * round(random.uniform(0, 1), len(str(n_samples)))
            X_new_1 = np.vstack((X_new_zhou, X_new_1))

    X_new_1 = np.delete(X_new_1, -1, 0)
    return X_new_1.astype(X.dtype)


class BaseSMOTE(BaseOverSampler):
    """Base class for the different SMOTE algorithms."""
    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=10,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs


    def _validate_estimator(self):
        """Check the NN estimators shared across the different SMOTE
        algorithms.
        """
        self.nn_k_ = check_neighbors_object(
            "k_neighbors", self.k_neighbors, additional_neighbor=1
        )
        self.nn_k_.set_params(**{"n_jobs": self.n_jobs})


    def _in_danger_noise(
        self, nn_estimator, samples, target_class, y, kind="danger"
    ):
        """Estimate if a set of sample are in danger or noise.
        Parameters
        ----------
        nn_estimator :  estimator
                        An estimator that inherits from
                        :class:`sklearn.neighbors.base.KNeighborsMixin` use to determine if
                        a sample is in danger/noise.
        samples :       {array-like, sparse matrix} of shape (n_samples, n_features)
                        The samples to check if either they are in danger or not.
        target_class :  int or str
                        The target corresponding class being over-sampled.
        y :             array-like of shape (n_samples,)
                        The true label in order to check the neighbour labels.
        kind :          {'danger', 'noise'}, default='danger'
                        The type of classification to use. Can be either:
                        - If 'danger', check if samples are in danger,
                        - If 'noise', check if samples are noise.
        Returns
        -------
        output :    ndarray of shape (n_samples,)
                    A boolean array where True refer to samples in danger or noise.
        """
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
        else:
            raise NotImplementedError


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class BorderlineSMOTE(BaseSMOTE):
    """
        Over-sampling using Borderline SMOTE.
        Parameters
        ----------
        {sampling_strategy}
        {random_state}
        k_neighbors : int or object, default=5
            If ``int``, number of nearest neighbours to used to construct synthetic
            samples.  If object, an estimator that inherits from
            :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
            find the k_neighbors.
        {n_jobs}
        m_neighbors : int or object, default=10
            If int, number of nearest neighbours to use to determine if a minority
            sample is in danger. If object, an estimator that inherits
            from :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used
            to find the m_neighbors.
    """

    '''
    nn_m是用来判断边界点，危险点，安全点,       m_neighbors=10,  继承自Base类
    nn_k 是求少数类点中的k近邻点            k_neighbors=5,
    '''
    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,      #nn_k, K近邻的参数
        n_jobs=None,
        m_neighbors=10,     #nn_m
        weight = None,
        ntree = 10,     #树的棵树，默认为10用来计算权重
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.m_neighbors = m_neighbors
        self.k_neighbors = k_neighbors
        self.weight = weight
        self.ntree = ntree

    def _validate_estimator(self):
        super()._validate_estimator()
        self.nn_m_ = check_neighbors_object(
            "m_neighbors", self.m_neighbors, additional_neighbor=1    
        )
        self.nn_m_.set_params(**{"n_jobs": self.n_jobs})


    def _fit_resample(self, X, y):
        self._validate_estimator()
        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:continue

            target_class_indices = np.flatnonzero(y == class_sample)    #返回true或1的index(少数点的索引)
            X_class = _safe_indexing(X, target_class_indices)       #少数类点的特征信息
            weight_maj = _safe_indexing(self.weight,target_class_indices)       #原始crf权重
            new_n_maj = [round((1-i/self.ntree),2) for i in weight_maj] #计算后的权重

            self.nn_m_.fit(X)           #nn_m是用来判断边界点，危险点，安全点
            danger_index ,n_maj= self._in_danger_noise(
                self.nn_m_, X_class, class_sample, y, kind="danger")
            if not any(danger_index):continue
        
            self.nn_k_.fit(X_class)     #训练KNN模型
            nns = self.nn_k_.kneighbors(    #边界点在少数类点中的近邻点
                _safe_indexing(X_class, danger_index), return_distance=False)[:, 1:]    

            X_new, y_new = make_samples_zhou(
                X=_safe_indexing(X_class, danger_index),          #原始boderlin_1只在边界点附近插值
                y_dtype=y.dtype,            #插入点的数据类型
                y_type=class_sample,       #需要插的样本类
                nn_data=X_class,            #所有少数类点的特征信息                    
                nn_num=nns,                #K近邻点的索引
                n_samples=n_samples,          #要生成的样本数                   
                new_n_maj=new_n_maj,        #权重矩阵
                danger_and_safe=np.count_nonzero(danger_index),   #母节点数量
                mother_point=np.flatnonzero(danger_index),      ##母节点的索引
            )
            #结合新旧坐标点
            if sparse.issparse(X_new): X_resampled = sparse.vstack([X_resampled, X_new])#判断是否是稀疏矩阵       
            else:X_resampled = np.vstack((X_resampled, X_new))    
            y_resampled = np.hstack((y_resampled, y_new))
            return  X_resampled,y_resampled


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SVMSMOTE(BaseSMOTE):
    """
    ----------
    {sampling_strategy}
    {random_state}
    k_neighbors : int or object, default=5
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.
    {n_jobs}
    m_neighbors : int or object, default=10
        If int, number of nearest neighbours to use to determine if a minority
        sample is in danger. If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the m_neighbors.
    svm_estimator : object, default=SVC()
        A parametrized :class:`sklearn.svm.SVC` classifier can be passed.
    out_step : float, default=0.5
        Step size when extrapolating.
    """

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
        m_neighbors=5,      
        svm_estimator=None,
        out_step=0.5,
        weight = None,
        ntree=10,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.m_neighbors = m_neighbors
        self.svm_estimator = svm_estimator
        self.out_step = out_step
        self.weight = weight        #传进来的crf权重矩阵
        self.ntree=ntree

    def _validate_estimator(self):
        super()._validate_estimator()
        self.nn_m_ = check_neighbors_object(
            "m_neighbors",self.m_neighbors, additional_neighbor=1     
        )
        self.nn_m_.set_params(**{"n_jobs": self.n_jobs})

        if self.svm_estimator is None:
            self.svm_estimator_ = SVC(
                gamma="scale", random_state=self.random_state
            )
        elif isinstance(self.svm_estimator, SVC):
            self.svm_estimator_ = clone(self.svm_estimator)
        else:
            raise_isinstance_error("svm_estimator", [SVC], self.svm_estimator)


    def _fit_resample(self, X, y):
        self._validate_estimator()
        random_state = check_random_state(self.random_state)
        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0: continue

            target_class_indices = np.flatnonzero(y == class_sample)#是在总体数据集X里面的所有少数类的索引
            X_class = _safe_indexing(X, target_class_indices)   #当前少数类的所有特征信息
            weight_maj = _safe_indexing(self.weight,target_class_indices)       #原始crf权重
            new_n_maj = [round((1-i/self.ntree),2) for i in weight_maj] #计算后的权重

            #求出少数类中的支持向量
            self.svm_estimator_.fit(X, y)
            support_index = self.svm_estimator_.support_[           #np.ndarray
                y[self.svm_estimator_.support_] == class_sample]    # 少数点支持向量在总数据集中的索引
            support_vector = _safe_indexing(X, support_index)       #少数类支持向量的坐标

            #求出少数类中支持向量中的噪声点索引,然后删除噪声点索引
            self.nn_m_.fit(X)           #求噪声点的最近邻模型要用全部点来训练
            noise_bool = self._in_danger_noise(
                self.nn_m_, support_vector, class_sample, y, kind="noise")[0] 
            noise_index = np.flatnonzero(noise_bool)
            support_index = np.delete(support_index,noise_index)    #删除少数累支持向量中的噪声点索引

            #取出支持向量中的噪声点，只剩安全点和边界点的特征信息
            support_vector = _safe_indexing(
                support_vector, np.flatnonzero(np.logical_not(noise_bool)))       
            
            #求出支持向量点中的边界点和安全点   在总数据集中的索引 
            danger_bool = self._in_danger_noise(
                self.nn_m_, support_vector, class_sample, y, kind="danger")[0]      
            safety_bool = np.logical_not(danger_bool)       #逻辑非，取反
            danger_index = np.delete(support_index,np.flatnonzero(safety_bool))
            safe_index = np.delete(support_index,np.flatnonzero(danger_bool))

            #求出支持向量点中的边界点和安全点   在少数类点的索引
            danger_list,safe_list = [],[]
            for i in danger_index:
                ii = np.where(target_class_indices == i)
                danger_list.append(ii[0][0])
            for i in safe_index:
                ii = np.where(target_class_indices == i)
                safe_list.append(ii[0][0])

            self.nn_k_.fit(X_class)     #用当前少数类的所有点来训练最近邻矩阵
            fractions = random_state.beta(10, 10)
            n_generated_samples = int(fractions * (n_samples + 1))

            #基于边界点生成的点
            if np.count_nonzero(danger_bool) > 0:
                #边界点在所有少数类点中的近邻点列表,默认按距离从近到远
                nns = self.nn_k_.kneighbors(
                    _safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    return_distance=False,)[:, 1:]  
                X_new_1, y_new_1 = make_samples_zhou(
                    X=_safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    y_dtype=y.dtype,
                    y_type=class_sample,
                    nn_data=X_class,                #当前少数类的所有特征信息
                    nn_num=nns,                     #近邻点的索引
                    n_samples=n_generated_samples,      #要生成的样本数
                    new_n_maj=new_n_maj,            #权重矩阵
                    danger_and_safe=len(danger_index),      #危险点或安全点的个数
                    mother_point = np.array(danger_list),   #种子节点的索引
                )

            #基于安全点生成的点
            if np.count_nonzero(safety_bool) > 0:
                nns = self.nn_k_.kneighbors(
                    _safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    return_distance=False,)[:, 1:]
                X_new_2, y_new_2 = make_samples_zhou(
                    X=_safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    y_dtype=y.dtype,
                    y_type=class_sample,
                    nn_data=X_class,
                    nn_num=nns,
                    n_samples=n_samples - n_generated_samples,
                    new_n_maj=new_n_maj,
                    danger_and_safe=len(safe_index),
                    mother_point=np.array(safe_list),
                )

            if (
                np.count_nonzero(danger_bool) > 0
                and np.count_nonzero(safety_bool) > 0
            ):
                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack(
                        [X_resampled, X_new_1, X_new_2]
                    )
                else:
                    X_resampled = np.vstack((X_resampled, X_new_1, X_new_2))
                y_resampled = np.concatenate(
                    (y_resampled, y_new_1, y_new_2), axis=0
                )
            elif np.count_nonzero(danger_bool) == 0:
                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack([X_resampled, X_new_2])
                else:
                    X_resampled = np.vstack((X_resampled, X_new_2))
                y_resampled = np.concatenate((y_resampled, y_new_2), axis=0)
            elif np.count_nonzero(safety_bool) == 0:
                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack([X_resampled, X_new_1])
                else:
                    X_resampled = np.vstack((X_resampled, X_new_1))
                y_resampled = np.concatenate((y_resampled, y_new_1), axis=0)
        return X_resampled, y_resampled


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SMOTE(BaseSMOTE):
    """Class to perform over-sampling using SMOTE.
    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique as presented in [1]_.
    Parameters
    ----------
    {sampling_strategy}
    {random_state}
    k_neighbors : int or object, default=5
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.
    {n_jobs}
    """

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
        weight = None,
        ntree = 10,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.weight=weight
        self.ntree=ntree

    def _fit_resample(self, X, y):
        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:continue    #需要合成的点数量

            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)   #当前少数类的特征信息
            weight_maj = _safe_indexing(self.weight,target_class_indices)       #原始crf权重
            new_n_maj = [round((1-i/self.ntree),2) for i in weight_maj] #计算后的权重

            self.nn_k_.fit(X_class)     #训练KNN模型，获取近邻点矩阵
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]

            X_new, y_new = make_samples_zhou(
                X=X_class,
                y_dtype=y.dtype,
                y_type=class_sample, 
                nn_data=X_class, 
                nn_num=nns, 
                n_samples=n_samples,
                new_n_maj=new_n_maj,
                danger_and_safe=len(new_n_maj), #种子样本的数量
                mother_point=np.arange(len(target_class_indices)),      #母节点在少数类中的索引
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        return X_resampled, y_resampled


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class KMeansSMOTE(BaseSMOTE):
    """Apply a KMeans clustering before to over-sample using SMOTE.
    This is an implementation of the algorithm described in [1]_.
    Read more in the :ref:`User Guide <smote_adasyn>`.
    Parameters
    ----------
    {sampling_strategy}
    {random_state}
    k_neighbors : int or object, default=2
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.
    {n_jobs}
    kmeans_estimator : int or object, default=None
        A KMeans instance or the number of clusters to be used. By default,
        we used a :class:`sklearn.cluster.MiniBatchKMeans` which tend to be
        better with large number of samples.
    cluster_balance_threshold : "auto" or float, default="auto"
        The threshold at which a cluster is called balanced and where samples
        of the class selected for SMOTE will be oversampled. If "auto", this
        will be determined by the ratio for each class, or it can be set
        manually.
    density_exponent : "auto" or float, default="auto"
        This exponent is used to determine the density of a cluster. Leaving
        this to "auto" will use a feature-length based exponent.
    Attributes
    ----------
    kmeans_estimator_ : estimator
        The fitted clustering method used before to apply SMOTE.
    nn_k_ : estimator
        The fitted k-NN estimator used in SMOTE.
    cluster_balance_threshold_ : float
        The threshold used during ``fit`` for calling a cluster balanced.
    """

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=2,              
        n_jobs=None,
        kmeans_estimator=None,
        cluster_balance_threshold="auto",
        density_exponent="auto",
        weight=None,
        ntree=10,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.kmeans_estimator = kmeans_estimator
        self.cluster_balance_threshold = cluster_balance_threshold
        self.density_exponent = density_exponent
        self.k_neighbors = k_neighbors
        self.weight=weight
        self.ntree=ntree

    def _validate_estimator(self,n_clusters_zhou):
        super()._validate_estimator()           #继承nn_k_
        if self.kmeans_estimator is None:
            self.kmeans_estimator_ = MiniBatchKMeans(
                # n_clusters=20,          #TODO默认是8
                random_state=self.random_state,
            )
        elif isinstance(self.kmeans_estimator, int):
            self.kmeans_estimator_ = MiniBatchKMeans(
                n_clusters=self.kmeans_estimator,
                random_state=self.random_state,)
        else:
            self.kmeans_estimator_ = clone(self.kmeans_estimator)       #克隆模型

        # validate the parameters
        for param_name in ("cluster_balance_threshold", "density_exponent"):
            param = getattr(self, param_name)
            if isinstance(param, str) and param != "auto":
                raise ValueError(
                    "'{}' should be 'auto' when a string is passed. "
                    "Got {} instead.".format(param_name, repr(param))
                )
        self.cluster_balance_threshold_ = (
            self.cluster_balance_threshold
            if self.kmeans_estimator_.n_clusters != 1
            else -np.inf
        )


    def _find_cluster_sparsity(self, X):
        """Compute the cluster sparsity."""
        euclidean_distances = pairwise_distances(
            X, metric="euclidean", n_jobs=self.n_jobs)
        # negate diagonal elements
        for ind in range(X.shape[0]):euclidean_distances[ind, ind] = 0

        non_diag_elements = (X.shape[0] ** 2) - X.shape[0]
        mean_distance = euclidean_distances.sum() / non_diag_elements
        exponent = (
            math.log(X.shape[0], 1.6) ** 1.8 * 0.16
            if self.density_exponent == "auto"
            else self.density_exponent)
        return (mean_distance ** exponent) / X.shape[0]


    def _fit_resample(self, X, y):
        X_resampled = X.copy()
        y_resampled = y.copy()

        if len(X_resampled)<100:n_clusters_zhou =5
        elif  len(X_resampled)<500:n_clusters_zhou = 8
        elif len(X_resampled) <1000:n_clusters_zhou = 15
        else: n_clusters_zhou = 30

        self._validate_estimator(n_clusters_zhou=n_clusters_zhou)
        total_inp_samples = sum(self.sampling_strategy_.values())
        # print('簇心数量:\t',self.kmeans_estimator_.n_clusters)
        
        for class_sample, n_samples in self.sampling_strategy_.items():#也适用于多分类
            '''Step_1: 聚类'''
            if n_samples == 0:continue  #不需要插值就跳过，插下一类
            X_clusters = self.kmeans_estimator_.fit_predict(X)  #聚类并返回对每个点预测结果(标签)
            valid_clusters = []     #筛选出来的簇
            cluster_sparsities = []

            '''Step_2: 过滤选择用于采样的簇，选择少数类多的簇,阈值0.5'''
            for cluster_idx in range(self.kmeans_estimator_.n_clusters):        #遍历每个簇
                cluster_mask = np.flatnonzero(X_clusters == cluster_idx)    #这个簇的索引
                X_cluster = _safe_indexing(X, cluster_mask)     #这个簇的点的横纵坐标
                y_cluster = _safe_indexing(y, cluster_mask)     #这个簇的点的标签
                cluster_class_mean = (y_cluster == class_sample).mean()     #少数类的占比，用来和阈值比较

                if self.cluster_balance_threshold_ == "auto":       #阈值，默认为0.5
                    balance_threshold = n_samples / total_inp_samples / 2       #TODO
                    # balance_threshold = 0.3
                else:balance_threshold = self.cluster_balance_threshold_        

                # the cluster is already considered balanced
                if cluster_class_mean < balance_threshold:continue #少数类比例<阈值

                # not enough samples to apply SMOTE
                anticipated_samples = cluster_class_mean * X_cluster.shape[0]
                if anticipated_samples < self.nn_k_.n_neighbors:continue

                X_cluster_class = _safe_indexing(   #筛选出当前簇里面是要添加的那种类的点(少数点)
                    X_cluster, np.flatnonzero(y_cluster == class_sample))

                valid_clusters.append(cluster_mask)
                cluster_sparsities.append(
                    self._find_cluster_sparsity(X_cluster_class)
                )

            cluster_sparsities = np.array(cluster_sparsities)
            cluster_weights = cluster_sparsities / cluster_sparsities.sum()

            if not valid_clusters:          #如果为空,报异常
                print('没有valid_clusters',valid_clusters,class_sample,'------------------------------------------------------------------------------')
                raise RuntimeError(
                    "No clusters found with sufficient samples of "
                    "class {}. Try lowering the cluster_balance_threshold "
                    "or increasing the number of "
                    "clusters.".format(class_sample)
                )

            '''Step_3: 对每个筛选出来的簇进行过采样'''
            for valid_cluster_idx, valid_cluster in enumerate(valid_clusters):      #分簇和对应的标签
                X_cluster = _safe_indexing(X, valid_cluster)        #找到筛选出来的簇的横纵坐标
                y_cluster = _safe_indexing(y, valid_cluster)        #筛选出来簇的标签
                weight_maj = _safe_indexing(self.weight,valid_cluster) #根据索引找到当前簇的原始crf权重
                new_n_maj = [round((1-i/self.ntree),2) for i in weight_maj] #计算后的权重

                X_cluster_class = _safe_indexing(       #去除簇里面的多数类点
                    X_cluster, np.flatnonzero(y_cluster == class_sample))
                new_n_maj = _safe_indexing(new_n_maj,np.flatnonzero(y_cluster == class_sample))

                #计算每个簇里要插的个数:cluster_n_samples
                if (n_samples * cluster_weights[valid_cluster_idx])%1 >=0.5 :
                    cluster_n_samples = int(
                        math.ceil(n_samples * cluster_weights[valid_cluster_idx]))
                elif (n_samples * cluster_weights[valid_cluster_idx])%1 <0.5:
                    cluster_n_samples = int(
                        math.floor(n_samples * cluster_weights[valid_cluster_idx]))

                self.nn_k_.fit(X_cluster_class)     #训练KNNmo'x
                nns = self.nn_k_.kneighbors(            #近邻点的下标
                    X_cluster_class, return_distance=False)[:, 1:]

                if cluster_n_samples !=0:
                    X_new, y_new = make_samples_zhou(
                        X=X_cluster_class,
                        y_dtype=y.dtype,
                        y_type=class_sample,
                        nn_data=X_cluster_class,
                        nn_num=nns,
                        n_samples=cluster_n_samples,        #这个簇里面需要插的个数
                        new_n_maj=new_n_maj,
                        danger_and_safe=len(new_n_maj),     #种子节点的数量
                        mother_point=np.arange(len(new_n_maj)), #中子节点在少数类中的索引
                    )
                X_resampled = np.vstack((X_resampled,X_new))
                y_resampled = np.hstack((y_resampled, y_new))
        return X_resampled, y_resampled
