# -*- coding: utf-8 -*-
'''crf-weight版本
POST-process
SMOTE-ENN,SMOTE-IPF,SMOTE-RSB,SMOTE_TomekLinks
Adasyn(PRE-process)
'''


import os
import pickle
import itertools
import logging
import re
import time
import glob
import inspect

# used to parallelize evaluation
from joblib import Parallel, delayed

# numerical methods and arrays
import numpy as np
import pandas as pd

# import packages used for the implementation of sampling methods
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_score, StratifiedKFold
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
#from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone, BaseEstimator, ClassifierMixin

# some statistical methods
from scipy.stats import skew
import scipy.signal as ssignal
import scipy.spatial as sspatial
import scipy.optimize as soptimize
import scipy.special as sspecial
from scipy.stats.mstats import gmean

# self-organizing map implementation
import minisom



__author__= "György Kovács"
__license__= "MIT"
__email__= "gyuriofkovacs@gmail.com"
__version__ = '0.3.4'
#for handler in _logger.root.handlers[:]:
#    _logger.root.removeHandler(handler)

# setting the _logger format
#_logger.basicConfig(filename= '/home/gykovacs/workspaces/sampling2.log', level= logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")
# _logger= logging.getLogger('smote_variants')
# _logger.setLevel(logging.DEBUG)
# _logger_ch= logging.StreamHandler()
# _logger_ch.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
# _logger.addHandler(_logger_ch)



__all__=[
        # 'SMOTE',
        # 'Safe_Level_SMOTE',
        # 'MWMOTE',
        'SMOTE_RSB',              #基于smote
        # 'ADASYN',                  #改好了
        'SMOTE_TomekLinks',         #基于SMOTE改的
        # 'Edge_Det_SMOTE',
        'SMOTE_ENN',                #基于smote
        # 'LN_SMOTE',
        # 'NRSBoundary_SMOTE',
        # 'cluster_SMOTE',
        'SMOTE_IPF',            #基于smote改的
        # 'SMOTE_PSO',
        # 'TRIM_SMOTE',
        ]


def get_all_oversamplers(all=__all__):
    """
        Returns all oversampling classes
        
        Returns:
            list(OverSampling): list of all oversampling classes
            
        Example::
            
            import smote_variants as sv
            
            oversamplers= sv.get_all_oversamplers()
    """
    return [globals()[s] for s in all if s in globals() and inspect.isclass(globals()[s]) and issubclass(globals()[s], OverSampling) and not globals()[s] == OverSampling]

def get_n_quickest_oversamplers(n= 10):
    """
    Returns the n quickest oversamplers based on testing on the datasets of
    the imbalanced_databases package.
    
    Args:
        n (int): number of oversamplers to return
    
    Returns:
        list(OverSampling): list of the n quickest oversampling classes
        
    Example::
        
        import smote_variants as sv
        
        oversamplers= sv.get_n_quickest_oversamplers(10)
    """
    
    runtimes= {'SPY': 0.11, 'OUPS': 0.16, 'SMOTE_D': 0.20, 'NT_SMOTE': 0.20, 'Gazzah': 0.21,
        'ROSE': 0.25, 'NDO_sampling': 0.27, 'Borderline_SMOTE1': 0.28, 'SMOTE': 0.28,
        'Borderline_SMOTE2': 0.29, 'ISMOTE': 0.30, 'SMMO': 0.31, 'SMOTE_OUT': 0.37,
        'SN_SMOTE': 0.44, 'Selected_SMOTE': 0.47, 'distance_SMOTE': 0.47, 'Gaussian_SMOTE': 0.48,
        'MCT': 0.51, 'Random_SMOTE': 0.57, 'ADASYN': 0.58, 'SL_graph_SMOTE': 0.58,
        'CURE_SMOTE': 0.59, 'ANS': 0.63, 'MSMOTE': 0.72, 'Safe_Level_SMOTE': 0.79,
        'SMOBD': 0.80, 'CBSO': 0.81, 'Assembled_SMOTE': 0.82, 'SDSMOTE': 0.88,
        'SMOTE_TomekLinks': 0.91, 'Edge_Det_SMOTE': 0.94, 'ProWSyn': 1.00, 'Stefanowski': 1.04,
        'NRAS': 1.06, 'AND_SMOTE': 1.13, 'DBSMOTE': 1.17, 'polynom_fit_SMOTE': 1.18,
        'ASMOBD': 1.18, 'MDO': 1.18, 'SOI_CJ': 1.24, 'LN_SMOTE': 1.26, 'VIS_RST': 1.34,
        'TRIM_SMOTE': 1.36, 'LLE_SMOTE': 1.62, 'SMOTE_ENN': 1.86, 'SMOTE_Cosine': 2.00,
        'kmeans_SMOTE': 2.43, 'MWMOTE': 2.45, 'V_SYNTH': 2.59, 'A_SUWO': 2.81,
        'RWO_sampling': 2.91, 'SMOTE_RSB': 3.88, 'ADOMS': 3.89, 'SMOTE_IPF': 4.10,
        'Lee': 4.16, 'SMOTE_FRST_2T': 4.18, 'cluster_SMOTE': 4.19, 'SOMO': 4.30,
        'DE_oversampling': 4.67, 'CCR': 4.72, 'NRSBoundary_SMOTE': 5.26, 'AHC': 5.27,
        'ISOMAP_Hybrid': 6.11, 'LVQ_SMOTE': 6.99, 'CE_SMOTE': 7.45, 'MSYN': 11.92,
        'PDFOS': 15.14, 'KernelADASYN': 17.87, 'G_SMOTE': 19.23, 'E_SMOTE': 19.50,
        'SVM_balance': 24.05, 'SUNDO': 26.21, 'GASMOTE': 31.38, 'DEAGO': 33.39,
        'NEATER': 41.39, 'SMOTE_PSO': 45.12, 'IPADE_ID': 90.01, 'DSMOTE': 146.73,
        'MOT2LD': 149.42, 'Supervised_SMOTE': 195.74, 'SSO': 215.27, 'DSRBF': 272.11,
        'SMOTE_PSOBAT': 324.31, 'ADG': 493.64, 'AMSCO': 1502.36}
    
    samplers= get_all_oversamplers()
    samplers= sorted(samplers, key= lambda x: runtimes[x.__name__] if x.__name__ in runtimes else 1e8)
    
    return samplers[:n]

def get_all_oversamplers_multiclass(strategy= "equalize_1_vs_many_successive"):
    """
    Returns all oversampling classes which can be used with the multiclass strategy specified
    
    Args:
        strategy (str): the multiclass oversampling strategy - 'equalize_1_vs_many_successive'/'equalize_1_vs_many'
    
    Returns:
        list(OverSampling): list of all oversampling classes which can be used with the multiclass strategy specified
        
    Example::
        
        import smote_variants as sv
        
        oversamplers= sv.get_all_oversamplers_multiclass()
    """
    
    oversamplers= get_all_oversamplers()
    
    if strategy == 'equalize_1_vs_many_successive' or strategy == 'equalize_1_vs_many':
        return [o for o in oversamplers if not OverSampling.cat_changes_majority in o.categories and 'proportion' in o().get_params()]
    else:
        raise ValueError("It is not known which oversamplers work with the strategy %s" % strategy)

def get_n_quickest_oversamplers_multiclass(n, strategy= "equalize_1_vs_many_successive"):
    """
    Returns the n quickest oversamplers based on testing on the datasets of
    the imbalanced_databases package, and suitable for using the multiclass strategy specified.
    
    Args:
        n (int): number of oversamplers to return
        strategy (str): the multiclass oversampling strategy - 'equalize_1_vs_many_successive'/'equalize_1_vs_many'
    
    Returns:
        list(OverSampling): list of n quickest oversampling classes which can be used with the multiclass strategy specified
        
    Example::
        
        import smote_variants as sv
        
        oversamplers= sv.get_n_quickest_oversamplers_multiclass()
    """
    
    oversamplers= get_all_oversamplers()
    quickest_oversamplers= get_n_quickest_oversamplers(len(oversamplers))
    
    if strategy == 'equalize_1_vs_many_successive' or strategy == 'equalize_1_vs_many':
        return [o for o in quickest_oversamplers if not OverSampling.cat_changes_majority in o.categories and 'proportion' in o().get_params()][:n]
    else:
        raise ValueError("It is not known which oversamplers work with the strategy %s" % strategy)

def get_all_noisefilters():
    """
    Returns all noise filters
    Returns:
        list(NoiseFilter): list of all noise filter classes
    """
    return [globals()[s] for s in __all__ if s in globals() and inspect.isclass(globals()[s]) and issubclass(globals()[s], NoiseFilter) and not globals()[s] == NoiseFilter]

def mode(data):
    values, counts= np.unique(data, return_counts= True)
    return values[np.where(counts == max(counts))[0][0]]

class StatisticsMixin:
    """
    Mixin to compute class statistics and determine minority/majority labels
    """
    def class_label_statistics(self, X, y):
        """
        determines class sizes and minority and majority labels
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        unique, counts= np.unique(y, return_counts= True)
        self.class_stats= dict(zip(unique, counts))
        self.minority_label= unique[0] if counts[0] < counts[1] else unique[1]
        self.majority_label= unique[1] if counts[0] < counts[1] else unique[0]

class RandomStateMixin:
    """
    Mixin to set random state
    """
    def set_random_state(self, random_state):
        """
        sets the random_state member of the object
        
        Args:
            random_state (int/np.random.RandomState/None): the random state initializer
        """
        
        self._random_state_init= random_state
        
        if random_state is None:
            self.random_state= np.random
        elif isinstance(random_state, int):
            self.random_state= np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state= random_state
        elif random_state is np.random:
            self.random_state= random_state
        else:
            raise ValueError("random state cannot be initialized by " + str(random_state))
        

class ParameterCheckingMixin:
    """
    Mixin to check if parameters come from a valid range
    """
    def check_in_range(self, x, name, r):
        """
        Check if parameter is in range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x < r[0] or x > r[1]:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s outside the range [%f,%f] is not allowed: %f" % (name, r[0], r[1], x))
    
    def check_out_range(self, x, name, r):
        """
        Check if parameter is outside of range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x >= r[0] and x <= r[1]:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s outside the range [%f,%f] is not allowed: %f" % (name, r[0], r[1], x))
    
    def check_less_or_equal(self, x, name, val):
        """
        Check if parameter is less than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x > val:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s greater than %f is not allowed: %f > %f" % (name, val, x, val))
    
    def check_less_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x > y:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s greater than parameter %s is not allowed: %f > %f" % (name_x, name_y, x, y))
    
    def check_less(self, x, name, val):
        """
        Check if parameter is less than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x >= val:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s greater than or equal to %f is not allowed: %f >= %f" % (name, val, x, val))
        
    def check_less_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x >= y:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s greater than or equal to parameter %s is not allowed: %f >= %f" % (name_x, name_y, x, y))
    
    def check_greater_or_equal(self, x, name, val):
        """
        Check if parameter is greater than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x < val:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s less than %f is not allowed: %f < %f" % (name, val, x, val))
        
    def check_greater_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x < y:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s less than parameter %s is not allowed: %f < %f" % (name_x, name_y, x, y))
    
    def check_greater(self, x, name, val):
        """
        Check if parameter is greater than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x <= val:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s less than or equal to %f is not allowed: %f < %f" % (name, val, x, val))
        
    def check_greater_par(self, x, name_x, y, name_y):
        """
        Check if parameter is greater than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x <= y:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s less than or equal to parameter %s is not allowed: %f <= %f" % (name_x, name_y, x, y))
    
    def check_equal(self, x, name, val):
        """
        Check if parameter is equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x == val:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s equal to parameter %f is not allowed: %f == %f" % (name, val, x, val))
        
    def check_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x == y:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s equal to parameter %s is not allowed: %f == %f" % (name_x, name_y, x, y))
    
    def check_isin(self, x, name, l):
        """
        Check if parameter is in list
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            l (list): list to check if parameter is in it
        Throws:
            ValueError
        """
        if not x in l:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter %s not in list %s is not allowed: %s" % (name, str(l), str(x)))
    
    def check_n_jobs(self, x, name):
        """
        Check n_jobs parameter
        Args:
            x (int/None): number of jobs
            name (str): the parameter name
        Throws:
            ValueError
        """
        if (x is None) or (not x is None and isinstance(x, int) and not x == 0):
            pass
        else:
            raise ValueError(self.__class__.__name__ + ": " + "Value for parameter n_jobs is not allowed: %s" % str(x))

class ParameterCombinationsMixin:
    """
    Mixin to generate parameter combinations
    """
    
    @classmethod
    def generate_parameter_combinations(cls, dictionary, num= None):
        """
        Generates reasonable paramter combinations
        Args:
            dictionary (dict): dictionary of paramter ranges
            num (int): maximum number of combinations to generate
        """
        combinations= [dict(zip(list(dictionary.keys()), p)) for p in list(itertools.product(*list(dictionary.values())))]
        if num is None:
            return combinations
        else:
            if hasattr(cls, 'random_state'):
                return cls.random_state.choice(combinations, num, replace= False)
            else:
                return np.random.choice(combinations, num, replace= False)

class NoiseFilter(StatisticsMixin, ParameterCheckingMixin, ParameterCombinationsMixin):
    """
    Parent class of noise filtering methods
    """
    def __init__(self):
        """
        Constructor
        """
        pass
        
    def remove_noise(self, X, y):
        """
        Removes noise
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        pass
    
    def get_params(self, deep=False):
        """
        Return parameters
        
        Returns:
            dict: dictionary of parameters
        """
        
        return {}
    
    def set_params(self, **params):
        """
        Set parameters
        
        Args:
            params (dict): dictionary of parameters
        """
        
        for key, value in params.items():
            setattr(self, key, value)
        
        return self
    
class TomekLinkRemoval(NoiseFilter):
    """
    Tomek link removal
    
    References:
        * BibTex::
            
            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    } 
    """
    def __init__(self, strategy= 'remove_majority', n_jobs= 1):
        """
        Constructor of the noise filter.
        
        Args:
            strategy (str): noise removal strategy: 'remove_majority'/'remove_both'
            n_jobs (int): number of jobs
        """
        super().__init__()
        
        self.check_isin(strategy, 'strategy', ['remove_majority', 'remove_both'])
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.strategy= strategy
        self.n_jobs= n_jobs
    
    def remove_noise(self, X, y):
        """
        Removes noise from dataset
        
        Args:
            X (np.matrix): features
            y (np.array): target labels
            
        Returns:
            np.matrix, np.array: dataset after noise removal
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)
        
        # using 2 neighbors because the first neighbor is the point itself
        nn= NearestNeighbors(n_neighbors= 2, n_jobs= self.n_jobs)
        distances, indices= nn.fit(X).kneighbors(X)
        
        # identify links
        links= []
        for i in range(len(indices)):
            if indices[indices[i][1]][1] == i:
                if not y[indices[i][1]] == y[indices[indices[i][1]][1]]:
                    links.append((i, indices[i][1]))
                    
        # determine links to be removed
        to_remove= []
        for l in links:
            if self.strategy == 'remove_majority':
                if y[l[0]] == self.minority_label:
                    to_remove.append(l[1])
                else:
                    to_remove.append(l[0])
            elif self.strategy == 'remove_both':
                to_remove.append(l[0])
                to_remove.append(l[1])
            else:
                raise ValueError(self.__class__.__name__ + ": " + 'No Tomek link removal strategy %s implemented' % self.strategy)
        
        to_remove= list(set(to_remove))
        
        return np.delete(X, to_remove, axis= 0), np.delete(y, to_remove)

class CondensedNearestNeighbors(NoiseFilter):
    """
    Condensed nearest neighbors
    
    References:
        * BibTex::
            
            @ARTICLE{condensed_nn, 
                        author={Hart, P.}, 
                        journal={IEEE Transactions on Information Theory}, 
                        title={The condensed nearest neighbor rule (Corresp.)}, 
                        year={1968}, 
                        volume={14}, 
                        number={3}, 
                        pages={515-516}, 
                        keywords={Pattern classification}, 
                        doi={10.1109/TIT.1968.1054155}, 
                        ISSN={0018-9448}, 
                        month={May}}
    """
    def __init__(self, n_jobs= 1):
        """
        Constructor of the noise removing object
        
        Args:
            n_jobs (int): number of jobs
        """
        super().__init__()
        
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.n_jobs= n_jobs
        
    def remove_noise(self, X, y):
        """
        Removes noise from dataset
        
        Args:
            X (np.matrix): features
            y (np.array): target labels
            
        Returns:
            np.matrix, np.array: dataset after noise removal
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)
        
        # Initial result set consists of all minority samples and 1 majority sample
        X_maj= X[y == self.majority_label]
        X_hat= np.vstack([X[y == self.minority_label], X_maj[0]])
        y_hat= np.hstack([np.repeat(self.minority_label, len(X_hat)-1), [self.majority_label]])
        X_maj= X_maj[1:]
        
        # Adding misclassified majority elements repeatedly        
        while True:
            knn= KNeighborsClassifier(n_neighbors= 1, n_jobs= self.n_jobs)
            knn.fit(X_hat, y_hat)
            pred= knn.predict(X_maj)
            
            if np.all(pred == self.majority_label):
                break
            else:
                X_hat= np.vstack([X_hat, X_maj[pred != self.majority_label]])
                y_hat= np.hstack([y_hat, np.repeat(self.majority_label, len(X_hat) - len(y_hat))])
                X_maj= np.delete(X_maj, np.where(pred != self.majority_label)[0], axis= 0)
                if len(X_maj) == 0:
                    break
        
        return X_hat, y_hat

class OneSidedSelection(NoiseFilter):
    """
    References:
        * BibTex::
            
            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    } 
    """
    def __init__(self, n_jobs= 1):
        """
        Constructor of the noise removal object
        
        Args:
            n_jobs (int): number of jobs
        """
        super().__init__()
        
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.n_jobs= n_jobs
        
    def remove_noise(self, X, y):
        """
        Removes noise
        
        Args:
            X (np.matrix): features
            y (np.array): target labels
            
        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)
        
        t= TomekLinkRemoval(n_jobs= self.n_jobs)
        X0, y0= t.remove_noise(X, y)
        cnn= CondensedNearestNeighbors(n_jobs= self.n_jobs)
        
        return cnn.remove_noise(X0, y0)

class CNNTomekLinks(NoiseFilter):
    """
    References:
        * BibTex::
            
            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    } 
    """
    def __init__(self, n_jobs= 1):
        """
        Constructor of the noise removal object
        
        Args:
            n_jobs (int): number of parallel jobs
        """
        super().__init__()
        
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.n_jobs= n_jobs
        
    def remove_noise(self, X, y):
        """
        Removes noise
        
        Args:
            X (np.matrix): features
            y (np.array): target labels
            
        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)
        
        c= CondensedNearestNeighbors(n_jobs= self.n_jobs)
        X0, y0= c.remove_noise(X, y)
        t= TomekLinkRemoval(n_jobs= self.n_jobs)
        
        return t.remove_noise(X0, y0)

class NeighborhoodCleaningRule(NoiseFilter):
    """
    References:
        * BibTex::
            
            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    } 
    """
    def __init__(self, n_jobs= 1):
        """
        Constructor of the noise removal object
        
        Args:
            n_jobs (int): number of parallel jobs
        """
        super().__init__()
        
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.n_jobs= n_jobs
        
    def remove_noise(self, X, y):
        """
        Removes noise
        
        Args:
            X (np.matrix): features
            y (np.array): target labels
            
        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)
        
        # fitting nearest neighbors with proposed parameter
        # using 4 neighbors because the first neighbor is the point itself
        nn= NearestNeighbors(n_neighbors= 4, n_jobs= self.n_jobs)
        nn.fit(X)
        distances, indices= nn.kneighbors(X)
        
        # identifying the samples to be removed
        to_remove= []
        for i in range(len(X)):
            if y[i] == self.majority_label and mode(y[indices[i][1:]]) == self.minority_label:
                # if sample i is majority and the decision based on neighbors is minority
                to_remove.append(i)
            elif y[i] == self.minority_label and mode(y[indices[i][1:]]) == self.majority_label:
                # if sample i is minority and the decision based on neighbors is majority
                for j in indices[i][1:]:
                    if y[j] == self.majority_label:
                        to_remove.append(j)
        
        # removing the noisy samples and returning the results
        to_remove= list(set(to_remove))
        return np.delete(X, to_remove, axis= 0), np.delete(y, to_remove)

class EditedNearestNeighbors(NoiseFilter):
    """
    References:
        * BibTex::
            
            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    } 
    """
    def __init__(self, remove= 'both', n_jobs= 1):
        """
        Constructor of the noise removal object
        
        Args:
            remove (str): class to remove from 'both'/'min'/'maj'
            n_jobs (int): number of parallel jobs
        """
        super().__init__()
        
        self.check_isin(remove, 'remove', ['both', 'min', 'maj'])
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.remove= remove
        self.n_jobs= n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise
        
        Args:
            X (np.matrix): features
            y (np.array): target labels
            
        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)
        
        nn= NearestNeighbors(n_neighbors= 4, n_jobs= self.n_jobs)
        nn.fit(X)
        distances, indices= nn.kneighbors(X)
        
        to_remove= []
        for i in range(len(X)):
            if not y[i] == mode(y[indices[i][1:]]):
                if self.remove == 'both' or (self.remove == 'min' and y[i] == self.minority_label) or (self.remove == 'maj' and y[i] == self.majority_label):
                    to_remove.append(i)
                
        return np.delete(X, to_remove, axis= 0), np.delete(y, to_remove)
    
    def get_params(self):
        """
        Get noise removal parameters
        
        Returns:
            dict: dictionary of parameters
        """
        return {'remove': self.remove}

class OverSampling(StatisticsMixin, ParameterCheckingMixin, ParameterCombinationsMixin, RandomStateMixin):
    """
    Base class of oversampling methods
    """
    
    categories= []
    cat_noise_removal= 'NR'
    cat_dim_reduction= 'DR'
    cat_uses_classifier= 'Clas'
    cat_sample_componentwise= 'SCmp'
    cat_sample_ordinary= 'SO'
    cat_sample_copy= 'SCpy'
    cat_memetic= 'M'
    cat_density_estimation= 'DE'
    cat_density_based= 'DB'
    cat_extensive= 'Ex'
    cat_changes_majority= 'CM'
    cat_uses_clustering= 'Clus'
    cat_borderline= 'BL'
    cat_application= 'A'
    
    def __init__(self):
        pass

    def number_of_instances_to_sample(self, strategy, n_maj, n_min):
        """
        Determines the number of samples to generate
        Args:
            strategy (str/float): if float, the fraction of the difference of
                                    the minority and majority numbers to generate, like
                                    0.1 means that 10% of the difference will be generated
                                    if str, like 'min2maj', the minority class will be upsampled
                                    to match the cardinality of the majority class
        """
        if isinstance(strategy, float) or isinstance(strategy, int):
            return max([0, int((n_maj - n_min)*strategy)])
        else:
            raise ValueError(self.__class__.__name__ + ": " + "Value %s for parameter strategy is not supported" % strategy)
    
    def sample_between_points(self, x, y):
        """
        Sample randomly along the line between two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
        Returns:
            np.array: the new sample
        """
        return x + (y - x)*self.random_state.random_sample()
    
    def sample_between_points_componentwise(self, x, y, mask= None):
        """
        Sample each dimension separately between the two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
            mask (np.array): array of 0,1s - specifies which dimensions to sample
        Returns:
            np.array: the new sample being generated
        """
        if mask is None:
            return x + (y - x)*self.random_state.random_sample()
        else:
            return x + (y - x)*self.random_state.random_sample()*mask
    
    def sample_by_jittering(self, x, std):
        """
        Sample by jittering.
        Args:
            x (np.array): base point
            std (float): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample() - 0.5)*2.0*std
    
    def sample_by_jittering_componentwise(self, x, std):
        """
        Sample by jittering componentwise.
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample(len(x))-0.5)*2.0 * std
    
    def sample_by_gaussian_jittering(self, x, std):
        """
        Sample by Gaussian jittering
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return self.random_state.normal(x, std)
    
    def sample(self, X, y):
        """
        The samplig function reimplemented in child classes
        Args:
            X (np.matrix): features
            y (np.array): labels
        Returns:
            np.matrix, np.array: sampled X and y
        """
        return X, y
    
    def sample_with_timing(self, X, y):
        begin= time.time()
        X_samp, y_samp= self.sample(X, y)
        # _logger.info(self.__class__.__name__ + ": " + ("runtime: %f" % (time.time() - begin)))
        return X_samp, y_samp
    
    def transform(self, X):
        """
        Transforms new data according to the possible transformation implemented
        by the function "sample".
        Args:
            X (np.matrix): features
        Returns:
            np.matrix: transformed features
        """
        return X
    
    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        pass
    
    def set_params(self, **params):
        """
        Set parameters
        
        Args:
            params (dict): dictionary of parameters
        """
        
        for key, value in params.items():
            setattr(self, key, value)
        
        return self
    
    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))
    
    def __str__(self):
        return self.descriptor()

class UnderSampling(StatisticsMixin, ParameterCheckingMixin, ParameterCombinationsMixin):
    """
    Base class of undersampling approaches.
    """
    def __init__(self):
        """
        Constructorm
        """
        super().__init__()
    
    def sample(self, X, y):
        """
        Carry out undersampling
        Args:
            X (np.matrix): features
            y (np.array): labels
        Returns:
            np.matrix, np.array: sampled X and y
        """
        pass
    
    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        pass
    
    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))



class SMOTE(OverSampling):
    
    categories= [OverSampling.cat_sample_ordinary,
                OverSampling.cat_extensive]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, n_jobs= 1, random_state= None):
        """
        Constructor of the SMOTE object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample e.g. 1.0 \
            means that after sampling the number of minority samples will be equal to the number of \
            majority samples
            n_neighbors (int): control parameter of the nearest neighbor technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
    
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7]})
                
    def sample(self, X, y,weight,ntree):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)

        if self.class_stats[self.minority_label] < 2:
            # _logger.warning(self.__class__.__name__ + ": " + "The number of minority samples (%d) is not enough for sampling" % self.class_stats[self.minority_label])
            return X.copy(), y.copy()
        
        # determining the number of samples to generate     #要插值的数量
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        X_min= X[y == self.minority_label]      #所有的少数类点
        # print('X_min:\t',type(X_min))
        
        # fitting the model
        n_neigh= min([len(X_min), self.n_neighbors+1])
        nn= NearestNeighbors(n_neighbors= n_neigh, n_jobs= self.n_jobs)
        nn.fit(X_min)
        dist, ind= nn.kneighbors(X_min)         #ind是每个所有少数类点的紧邻点索引
        
        if num_to_sample == 0:return X.copy(), y.copy()
        
        # generating samples
        base_indices= self.random_state.choice(list(range(len(X_min))), num_to_sample)
        neighbor_indices= self.random_state.choice(list(range(1, n_neigh)), num_to_sample)
        # print('base_indices:\t',len(base_indices),type(base_indices),base_indices,'\nneighbor_indices:\t',len(neighbor_indices))

        X_base= X_min[base_indices]     #随机选择num_to_sample个少数类点   base点
        X_neighbor= X_min[ind[base_indices, neighbor_indices]]  #每个base点的紧邻点,总共有num_to_sample个近邻点
        # print('X_base:\t',len(X_base),'\t\tX_neighbor:\t',len(X_neighbor))

        from crf_weight_api import add_weight
        samples = add_weight(X=X,
            y=y,
            X_min=X_min,
            minority_label=self.minority_label,
            base_indices=base_indices,
            neighbor_indices=neighbor_indices,
            num_to_sample=num_to_sample,
            ind=ind,
            X_neighbor=X_neighbor,
            X_base=X_base,
            weight=weight,
            ntree=ntree,
            )

        #递归补充新样本
        # if len(samples)<num_to_sample:
        #     # print('\n\n\n新样本数量:\t',len(samples),'\n\n\n')
        #     self.sample(np.vstack([X, samples]), np.hstack([y, np.hstack([self.minority_label]*len(samples))]),)

        return np.vstack([X, samples]), np.hstack([y, np.hstack([self.minority_label]*len(samples))])
    

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

class SMOTE_TomekLinks(OverSampling):
    
    categories= [OverSampling.cat_sample_ordinary,
            OverSampling.cat_noise_removal,
            OverSampling.cat_changes_majority]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, n_jobs= 1, random_state= None):
        """
        Constructor of the SMOTE object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to \
            sample e.g. 1.0 means that after sampling the number of minority samples \
            will be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return SMOTE.parameter_combinations()

    def sample(self, X, y,weight,ntree):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        smote= SMOTE(self.proportion, self.n_neighbors, n_jobs= self.n_jobs, random_state= self.random_state)
        X_new, y_new= smote.sample(X, y,weight,ntree)
        
        t= TomekLinkRemoval(strategy= 'remove_both', n_jobs= self.n_jobs)
        
        X_samp, y_samp= t.remove_noise(X_new, y_new)
        
        if len(X_samp) == 0:
            # _logger.info(self.__class__.__name__ + ": " + "All samples have been removed, returning original dataset.")
            return X.copy(), y.copy()
        
        return X_samp, y_samp
    
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
    
class SMOTE_ENN(OverSampling):
    """
    
    Notes:
        * Can remove too many of minority samples.
    """
    
    categories= [OverSampling.cat_sample_ordinary,
                 OverSampling.cat_noise_removal,
                 OverSampling.cat_changes_majority]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, n_jobs= 1, random_state= None):
        """
        Constructor of the SMOTE object
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
    
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return SMOTE.parameter_combinations()
    
    def sample(self, X, y,weight,ntree):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        self.class_label_statistics(X, y)
        
        if self.class_stats[self.minority_label] < 2:
            return X.copy(), y.copy()
        
        smote= SMOTE(self.proportion, self.n_neighbors, n_jobs= self.n_jobs, random_state=self.random_state)
        X_new, y_new= smote.sample(X, y,weight,ntree)
        enn= EditedNearestNeighbors(n_jobs= self.n_jobs)
        return enn.remove_noise(X_new, y_new)
    
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

class Borderline_SMOTE1(OverSampling):
    """
    References:
        * BibTex::
            
            @InProceedings{borderlineSMOTE,
                            author="Han, Hui
                            and Wang, Wen-Yuan
                            and Mao, Bing-Huan",
                            editor="Huang, De-Shuang
                            and Zhang, Xiao-Ping
                            and Huang, Guang-Bin",
                            title="Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning",
                            booktitle="Advances in Intelligent Computing",
                            year="2005",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="878--887",
                            isbn="978-3-540-31902-3"
                            }
    """
    
    categories= [OverSampling.cat_sample_ordinary,
                 OverSampling.cat_extensive,
                 OverSampling.cat_borderline]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, k_neighbors= 5, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor technique for determining the borderline
            k_neighbors (int): control parameter of the nearest neighbor technique for sampling
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(k_neighbors, 'k_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.k_neighbors= k_neighbors
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7], 
                                                    'k_neighbors': [3, 5, 7]})
    
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        if self.class_stats[self.minority_label] < 2:
            # _logger.warning(self.__class__.__name__ + ": " + "The number of minority samples (%d) is not enough for sampling" % self.class_stats[self.minority_label])
            return X.copy(), y.copy()
        
        # determining number of samples to be generated
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        # fitting model
        X_min= X[y == self.minority_label]
        
        nn= NearestNeighbors(self.n_neighbors+1, n_jobs= self.n_jobs)
        nn.fit(X)
        distances, indices= nn.kneighbors(X_min)
        
        # determining minority samples in danger
        noise= []
        danger= []
        for i in range(len(indices)):
            if self.n_neighbors == sum(y[indices[i][1:]] == self.majority_label):
                noise.append(i)
            elif mode(y[indices[i][1:]]) == self.majority_label:
                danger.append(i)
        X_danger= X_min[danger]
        X_min= np.delete(X_min, np.array(noise), axis= 0)
        
        if self.class_stats[self.minority_label] < 2:
            # _logger.warning(self.__class__.__name__ + ": " + "The number of minority samples (%d) is not enough for sampling" % self.class_stats[self.minority_label])
            return X.copy(), y.copy()
        
        if len(X_danger) == 0:
            # _logger.info(self.__class__.__name__ + ": " + "No samples in danger")
            return X.copy(), y.copy()
        
        # fitting nearest neighbors model to minority samples
        k_neigh= min([len(X_min), self.k_neighbors + 1])
        nn= NearestNeighbors(k_neigh, n_jobs= self.n_jobs)
        nn.fit(X_min)
        # extracting neighbors of samples in danger
        distances, indices= nn.kneighbors(X_danger)
        
        # generating samples near points in danger
        base_indices= self.random_state.choice(list(range(len(X_danger))), num_to_sample)
        neighbor_indices= self.random_state.choice(list(range(1, k_neigh)), num_to_sample)
        
        X_base= X_danger[base_indices]
        X_neighbor= X_min[indices[base_indices, neighbor_indices]]
        
        samples= X_base + np.multiply(self.random_state.rand(num_to_sample, 1), X_neighbor - X_base)
        
        return np.vstack([X, samples]), np.hstack([y, np.hstack([self.minority_label]*num_to_sample)])
    
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'k_neighbors': self.k_neighbors, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
    
class Borderline_SMOTE2(OverSampling):
    """
    References:
        * BibTex::
            
            @InProceedings{borderlineSMOTE,
                            author="Han, Hui
                            and Wang, Wen-Yuan
                            and Mao, Bing-Huan",
                            editor="Huang, De-Shuang
                            and Zhang, Xiao-Ping
                            and Huang, Guang-Bin",
                            title="Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning",
                            booktitle="Advances in Intelligent Computing",
                            year="2005",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="878--887",
                            isbn="978-3-540-31902-3"
                            }
    """
    
    categories= [OverSampling.cat_sample_ordinary,
                 OverSampling.cat_extensive,
                 OverSampling.cat_borderline]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, k_neighbors= 5, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor technique for determining the borderline
            k_neighbors (int): control parameter of the nearest neighbor technique for sampling
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(k_neighbors, 'k_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.k_neighbors= k_neighbors
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7], 
                                                    'k_neighbors': [3, 5, 7]})
    
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        if self.class_stats[self.minority_label] < 2:
            # _logger.warning(self.__class__.__name__ + ": " + "The number of minority samples (%d) is not enough for sampling" % self.class_stats[self.minority_label])
            return X.copy(), y.copy()
        
        # determining number of samples to be generated
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        # fitting nearest neighbors model
        X_min= X[y == self.minority_label]
        
        nn= NearestNeighbors(self.n_neighbors+1, n_jobs= self.n_jobs)
        nn.fit(X)
        distances, indices= nn.kneighbors(X_min)
        
        # determining minority samples in danger
        noise= []
        danger= []
        for i in range(len(indices)):
            if self.n_neighbors == sum(y[indices[i][1:]] == self.majority_label):
                noise.append(i)
            elif mode(y[indices[i][1:]]) == self.majority_label:
                danger.append(i)
        X_danger= X_min[danger]
        X_min= np.delete(X_min, np.array(noise), axis= 0)
        
        if len(X_min) < 2:
            # _logger.warning(self.__class__.__name__ + ": " + "The number of minority samples (%d) is not enough for sampling" % self.class_stats[self.minority_label])
            return X.copy(), y.copy()
        
        if len(X_danger) == 0:
            # _logger.info(self.__class__.__name__ + ": " + "No samples in danger")
            return X.copy(), y.copy()
        
        # fitting nearest neighbors model to minority samples
        k_neigh= self.k_neighbors + 1
        nn= NearestNeighbors(k_neigh, n_jobs= self.n_jobs)
        nn.fit(X)
        distances, indices= nn.kneighbors(X_danger)
        
        # generating the samples
        base_indices= self.random_state.choice(list(range(len(X_danger))), num_to_sample)
        neighbor_indices= self.random_state.choice(list(range(1, k_neigh)), num_to_sample)
        
        X_base= X_danger[base_indices]
        X_neighbor= X[indices[base_indices, neighbor_indices]]
        diff= X_neighbor - X_base
        r= self.random_state.rand(num_to_sample, 1)
        r[y[neighbor_indices] == self.majority_label]= r[y[neighbor_indices] == self.majority_label]*0.5
        
        samples= X_base + np.multiply(r, diff)
        
        return np.vstack([X, samples]), np.hstack([y, np.hstack([self.minority_label]*num_to_sample)])
    
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'k_neighbors': self.k_neighbors, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

class ADASYN(OverSampling):
    
    categories= [OverSampling.cat_sample_ordinary,
                OverSampling.cat_extensive,
                OverSampling.cat_borderline,
                OverSampling.cat_density_based]
    
    def __init__(self, n_neighbors= 5, d_th= 0.9, beta= 1.0, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            n_neighbors (int): control parameter of the nearest neighbor component
            d_th (float): tolerated deviation level from balancedness
            beta (float): target level of balancedness, same as proportion in other techniques
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(d_th, 'd_th', 0)
        self.check_greater_or_equal(beta, 'beta', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.n_neighbors= n_neighbors
        self.d_th= d_th
        self.beta= beta
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'n_neighbors': [3, 5, 7, 9], 
                                                    'd_th': [0.9], 
                                                    'beta': [1.0, 0.75, 0.5, 0.25]})
    
    def sample(self, X, y,weight,ntree):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        if self.class_stats[self.minority_label] < 2:
            # _logger.warning(self.__class__.__name__ + ": " + "The number of minority samples (%d) is not enough for sampling" % self.class_stats[self.minority_label])
            return X.copy(), y.copy()
        
        # extracting minority samples
        X_min= X[y == self.minority_label]
        
        # checking if sampling is needed
        m_min= len(X_min)
        m_maj= len(X) - m_min
        
        num_to_sample= (m_maj - m_min)*self.beta
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        d= float(m_min)/m_maj
        if d > self.d_th:
            return X.copy(), y.copy()
        
        # fitting nearest neighbors model to all samples
        nn= NearestNeighbors(min([len(X_min), self.n_neighbors+1]), n_jobs= self.n_jobs)
        nn.fit(X)
        distances, indices= nn.kneighbors(X_min)
        
        # determining the distribution of points to be generated
        r= []
        for i in range(len(indices)):
            r.append(sum(y[indices[i][1:]] == self.majority_label)/self.n_neighbors)
        r= np.array(r)
        r= r/sum(r)
        
        if any(np.isnan(r)):
            # _logger.warning(self.__class__.__name__ + ": " + "not enough non-noise samples for oversampling")
            return X.copy(), y.copy()
        
        # fitting nearest neighbors models to minority samples
        n_neigh= min([len(X_min), self.n_neighbors + 1])
        nn= NearestNeighbors(n_neigh, n_jobs= self.n_jobs)
        nn.fit(X_min)
        distances, indices= nn.kneighbors(X_min)
        
        # sampling points
        base_indices= self.random_state.choice(list(range(len(X_min))), size=int(num_to_sample), p=r)
        neighbor_indices= self.random_state.choice(list(range(1, n_neigh)), int(num_to_sample))
        
        X_base= X_min[base_indices]
        X_neighbor= X_min[indices[base_indices, neighbor_indices]]
        # print(type(X_base),type(X_neighbor))
        # print(X_neighbor,X_base)
        diff= X_neighbor - X_base
        r= self.random_state.rand(int(num_to_sample), 1)
        
        from crf_weight_api import add_weight
        samples = add_weight(X=X,
            y=y,
            X_min=X_min,
            minority_label=self.minority_label,
            base_indices=base_indices,
            neighbor_indices=neighbor_indices,
            num_to_sample=num_to_sample,
            ind=indices,
            X_neighbor=X_neighbor,
            X_base=X_base,
            weight = weight,
            ntree=ntree
        )

        # samples= X_base + np.multiply(r, diff)


        #递归补充新样本
        if len(samples)<num_to_sample:
            print('\n\n\n新样本数量:\t',len(samples),'\n\n\n')
            self.sample(np.vstack([X, samples]), np.hstack([y, np.hstack([self.minority_label]*len(samples))]))

        # samples= X_base + np.multiply(self.random_state.rand(num_to_sample, 1), X_neighbor - X_base)
        elif len(samples) == num_to_sample: 
            print('X和Y长度相同:',len(samples))



        # return np.vstack([X, samples]), np.hstack([y, np.hstack([self.minority_label]*int(num_to_sample))])
        return np.vstack([X, samples]), np.hstack([y, np.hstack([self.minority_label]*len(samples))])
    
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors, 
                'd_th': self.d_th, 
                'beta': self.beta, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}



class Safe_Level_SMOTE(OverSampling):
    """   
    Notes:
        * The original method was not prepared for the case when no minority sample has minority neighbors.
    """
    
    categories= [OverSampling.cat_borderline,
                OverSampling.cat_extensive,
                OverSampling.cat_sample_componentwise]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1.0)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7]})
        
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        # determine the number of samples to generate
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        # fitting nearest neighbors model
        nn= NearestNeighbors(n_neighbors= min([self.n_neighbors+1, len(X)]), n_jobs= self.n_jobs)
        nn.fit(X)
        distance, indices= nn.kneighbors(X)
        
        minority_labels= (y == self.minority_label)
        minority_indices= np.where(minority_labels)[0]
        
        # do the sampling
        numattrs= len(X[0])
        samples= []
        for _ in range(num_to_sample):
            index= self.random_state.randint(len(minority_indices))
            neighbor_index= self.random_state.choice(indices[index][1:])
            
            p= X[index]
            n= X[neighbor_index]
            
            # find safe levels
            sl_p= np.sum(y[indices[index][1:]] == self.minority_label)
            sl_n= np.sum(y[indices[neighbor_index][1:]] == self.minority_label)
            
            if sl_n > 0:
                sl_ratio= float(sl_p)/sl_n
            else:
                sl_ratio= np.inf
            
            if sl_ratio == np.inf and sl_p == 0:
                pass
            else:
                s= np.zeros(numattrs)
                for atti in range(numattrs):
                    # iterate through attributes and do sampling according to 
                    # safe level
                    if sl_ratio == np.inf and sl_p > 0:
                        gap= 0.0
                    elif sl_ratio == 1:
                        gap= self.random_state.random_sample()
                    elif sl_ratio > 1:
                        gap= self.random_state.random_sample()*1.0/sl_ratio
                    elif sl_ratio < 1:
                        gap= (1 - sl_ratio) + self.random_state.random_sample()*sl_ratio
                    dif= n[atti] - p[atti]
                    s[atti]= p[atti] + gap*dif
                samples.append(s)
                
        if len(samples) == 0:
            # _logger.warning(self.__class__.__name__ + ": " +"No samples generated")
            return X.copy(), y.copy()
        else:
            return np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(self.minority_label, len(samples))])
    
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}



# Borrowed from sklearn-dev, will be removed once the sklearn implementation
# becomes stable
class OPTICS:
    def __init__(self, min_samples=5, max_eps=np.inf, metric='euclidean',
                 p=2, metric_params=None, maxima_ratio=.75,
                 rejection_ratio=.7, similarity_threshold=0.4,
                 significant_min=.003, min_cluster_size=.005,
                 min_maxima_ratio=0.001, algorithm='ball_tree',
                 leaf_size=30, n_jobs=1):

        self.max_eps = max_eps
        self.min_samples = min_samples
        self.maxima_ratio = maxima_ratio
        self.rejection_ratio = rejection_ratio
        self.similarity_threshold = similarity_threshold
        self.significant_min = significant_min
        self.min_cluster_size = min_cluster_size
        self.min_maxima_ratio = min_maxima_ratio
        self.algorithm = algorithm
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Perform OPTICS clustering
        Extracts an ordered list of points and reachability distances, and
        performs initial clustering using `max_eps` distance specified at
        OPTICS object instantiation.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data.
        y : ignored
        Returns
        -------
        self : instance of OPTICS
            The instance.
        """
        n_samples = len(X)

        if self.min_samples > n_samples:
            raise ValueError(self.__class__.__name__ + ": " + "Number of training samples (n_samples=%d) must "
                             "be greater than min_samples (min_samples=%d) "
                             "used for clustering." %
                             (n_samples, self.min_samples))

        if self.min_cluster_size <= 0 or (self.min_cluster_size !=
                                          int(self.min_cluster_size)
                                          and self.min_cluster_size > 1):
            raise ValueError(self.__class__.__name__ + ": " + 'min_cluster_size must be a positive integer or '
                             'a float between 0 and 1. Got %r' %
                             self.min_cluster_size)
        elif self.min_cluster_size > n_samples:
            raise ValueError(self.__class__.__name__ + ": " + 'min_cluster_size must be no greater than the '
                             'number of samples (%d). Got %d' %
                             (n_samples, self.min_cluster_size))

        # Start all points as 'unprocessed' ##
        self.reachability_ = np.empty(n_samples)
        self.reachability_.fill(np.inf)
        self.core_distances_ = np.empty(n_samples)
        self.core_distances_.fill(np.nan)
        # Start all points as noise ##
        self.labels_ = np.full(n_samples, -1, dtype=int)

        nbrs = NearestNeighbors(n_neighbors=self.min_samples,
                                algorithm=self.algorithm,
                                leaf_size=self.leaf_size, metric=self.metric,
                                metric_params=self.metric_params, p=self.p,
                                n_jobs=self.n_jobs)

        nbrs.fit(X)
        self.core_distances_[:] = nbrs.kneighbors(X,
                                                  self.min_samples)[0][:, -1]

        self.ordering_ = self._calculate_optics_order(X, nbrs)

        return self

    # OPTICS helper functions

    def _calculate_optics_order(self, X, nbrs):
        # Main OPTICS loop. Not parallelizable. The order that entries are
        # written to the 'ordering_' list is important!
        processed = np.zeros(X.shape[0], dtype=bool)
        ordering = np.zeros(X.shape[0], dtype=int)
        ordering_idx = 0
        for point in range(X.shape[0]):
            if processed[point]:
                continue
            if self.core_distances_[point] <= self.max_eps:
                while not processed[point]:
                    processed[point] = True
                    ordering[ordering_idx] = point
                    ordering_idx += 1
                    point = self._set_reach_dist(point, processed, X, nbrs)
            else:  # For very noisy points
                ordering[ordering_idx] = point
                ordering_idx += 1
                processed[point] = True
        return ordering

    def _set_reach_dist(self, point_index, processed, X, nbrs):
        P = X[point_index:point_index + 1]
        indices = nbrs.radius_neighbors(P, radius=self.max_eps,
                                        return_distance=False)[0]

        # Getting indices of neighbors that have not been processed
        unproc = np.compress((~np.take(processed, indices)).ravel(),
                             indices, axis=0)
        # Keep n_jobs = 1 in the following lines...please
        if not unproc.size:
            # Everything is already processed. Return to main loop
            return point_index

        dists = pairwise_distances(P, np.take(X, unproc, axis=0),
                                   self.metric, n_jobs=1).ravel()

        rdists = np.maximum(dists, self.core_distances_[point_index])
        new_reach = np.minimum(np.take(self.reachability_, unproc), rdists)
        self.reachability_[unproc] = new_reach

        # Define return order based on reachability distance
        return (unproc[self.quick_scan(np.take(self.reachability_, unproc),
                                  dists)])

    def isclose(self, a, b, rel_tol= 1e-09, abs_tol= 0.0):
        return abs(a-b) <= max([rel_tol*max([abs(a), abs(b)]), abs_tol])
    
    def quick_scan(self, rdists, dists):
        rdist= np.inf
        dist= np.inf
        n= len(rdists)
        for i in range(n):
            if rdists[i] < rdist:
                rdist= rdists[i]
                dist= dists[i]
                idx= i
            elif self.isclose(rdists[i], rdist):
                if dists[i] < dist:
                    dist= dists[i]
                    idx= i
        return idx


class TRIM_SMOTE(OverSampling):
    """
        Notes:
        * It is not described precisely how the filtered data is used for sample generation. The method is proposed to be a preprocessing step, and it states that it applies sample generation to each group extracted. 
    """
    
    categories= [OverSampling.cat_extensive,
                 OverSampling.cat_uses_clustering]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, min_precision= 0.3, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_in_range(min_precision, 'min_precision', [0,1])
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.min_precision= min_precision
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
    
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7], 
                                                    'min_precision': [0.3]})
    
    def trim(self, y):
        """
        Determines the trim value.
        
        Args:
            y (np.array): array of target labels
            
        Returns:
            float: the trim value
        """
        return np.sum(y == self.minority_label)**2/len(y)
    
    def precision(self, y):
        """
        Determines the precision value.
        
        Args:
            y (np.array): array of target labels
            
        Returns:
            float: the precision value
        """
        return np.sum(y == self.minority_label)/len(y)
    
    def determine_splitting_point(self, X, y, split_on_border= False):
        """
        Determines the splitting point.
        
        Args:
            X (np.matrix): a subset of the training data
            y (np.array): an array of target labels
            split_on_border (bool): wether splitting on class borders is considered
            
        Returns:
            tuple(int, float), bool: (splitting feature, splitting value), make the split
        """
        trim_value= self.trim(y)
        d= len(X[0])
        max_t_minus_gain= 0.0
        split= None
        
        # checking all dimensions of X
        for i in range(d):
            # sort the elements in dimension i
            sorted_X_y= sorted(zip(X[:,i], y), key= lambda pair: pair[0])
            sorted_y= [yy for _, yy in sorted_X_y]
            
            # number of minority samples on the left
            left_min= 0
            # number of minority samples on the right
            right_min= np.sum(sorted_y == self.minority_label)
            
            # check all possible splitting points sequentiall
            for j in range(0, len(sorted_y)-1):
                if sorted_y[j] == self.minority_label:
                    # adjusting the number of minority and majority samples
                    left_min= left_min + 1
                    right_min= right_min - 1
                # checking of we can split on the border and do not split tieing feature values
                if (split_on_border == False or (split_on_border == True and not sorted_y[j-1] == sorted_y[j])) and sorted_X_y[j][0] != sorted_X_y[j+1][0]:
                    # compute trim value of the left
                    trim_left= left_min**2/(j+1)
                    # compute trim value of the right
                    trim_right= right_min**2/(len(sorted_y) - j - 1)
                    # let's check the gain
                    if max([trim_left, trim_right]) > max_t_minus_gain:
                        max_t_minus_gain= max([trim_left, trim_right])
                        split= (i, sorted_X_y[j][0])
        # return splitting values and the value of the logical condition in line 9
        if not split is None:
            return split, max_t_minus_gain > trim_value
        else:
            return (0, 0), False
    
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        leafs= [(X, y)]
        candidates= []
        seeds= []
        
        # executing the trimming
        # loop in line 2 of the paper
        # _logger.info(self.__class__.__name__ + ": " +"do the trimming process")
        while len(leafs) > 0 or len(candidates) > 0:
            add_to_leafs= []
            # executing the loop starting in line 3
            for l in leafs:
                # the function implements the loop starting in line 6
                # splitting on class border is forced
                split, gain= self.determine_splitting_point(l[0], l[1], True)
                if len(l[0]) == 1:
                    # small leafs with 1 element (no splitting point) are dropped
                    # as noise
                    continue
                else:
                    # condition in line 9
                    if gain:
                        # making the split
                        mask_left= (l[0][:,split[0]] <= split[1])
                        X_left, y_left= l[0][mask_left], l[1][mask_left]
                        mask_right= np.logical_not(mask_left)
                        X_right, y_right= l[0][mask_right], l[1][mask_right]
                        
                        # condition in line 11
                        if np.sum(y_left == self.minority_label) > 0:
                            add_to_leafs.append((X_left, y_left))
                        # condition in line 13
                        if np.sum(y_right == self.minority_label) > 0:
                            add_to_leafs.append((X_right, y_right))
                    else:
                        # line 16
                        candidates.append(l)
            # we implement line 15 and 18 by replacing the list of leafs by
            # the list of new leafs.
            leafs= add_to_leafs

            # iterating through all candidates (loop starting in line 21)
            for c in candidates:
                # extracting splitting points, this time split on border is not forced
                split, gain= self.determine_splitting_point(l[0], l[1], False)
                if len(l[0]) == 1:
                    # small leafs are dropped as noise
                    continue
                else:
                    # checking condition in line 27
                    if gain:
                        # doing the split
                        mask_left= (c[0][:,split[0]] <= split[1])
                        X_left, y_left= c[0][mask_left], c[1][mask_left]
                        mask_right= np.logical_not(mask_left)
                        X_right, y_right= c[0][mask_right], c[1][mask_right]
                        # checking logic in line 29
                        if np.sum(y_left == self.minority_label) > 0:
                            leafs.append((X_left, y_left))
                        # checking logic in line 31
                        if np.sum(y_right == self.minority_label) > 0:
                            leafs.append((X_right, y_right))
                    else:
                        # adding candidate to seeds (line 35)
                        seeds.append(c)
            # line 33 and line 36 are implemented by emptying the candidates list
            candidates= []
        
        # filtering the resulting set
        filtered_seeds= [s for s in seeds if self.precision(s[1]) > self.min_precision]
        
        # handling the situation when no seeds were found
        if len(seeds) == 0:
            # _logger.warning(self.__class__.__name__ + ": " +"no seeds identified")
            return X.copy(), y.copy()
        
        # fix for bad choice of min_precision
        multiplier= 0.9
        while len(filtered_seeds) == 0:
            filtered_seeds= [s for s in seeds if self.precision(s[1]) > self.min_precision*multiplier]
            multiplier= multiplier*0.9
            if multiplier < 0.1:
                # _logger.warning(self.__class__.__name__ + ": " + "no clusters passing the filtering")
                return X.copy(), y.copy()

        seeds= filtered_seeds
        
        X_seed= np.vstack([s[0] for s in seeds])
        y_seed= np.hstack([s[1] for s in seeds])
        
        # _logger.info(self.__class__.__name__ + ": " +"do the sampling")
        # generating samples by SMOTE
        X_seed_min= X_seed[y_seed == self.minority_label]
        if len(X_seed_min) <= 1:
            # _logger.warning(self.__class__.__name__ + ": " + "X_seed_min contains less than 2 samples")
            return X.copy(), y.copy()
        
        nn= NearestNeighbors(n_neighbors= min([len(X_seed_min), self.n_neighbors+1]), n_jobs= self.n_jobs)
        nn.fit(X_seed_min)
        distances, indices= nn.kneighbors(X_seed_min)
        
        # do the sampling
        samples= []
        for _ in range(num_to_sample):
            random_idx= self.random_state.randint(len(X_seed_min))
            random_neighbor_idx= self.random_state.choice(indices[random_idx][1:])
            samples.append(self.sample_between_points(X_seed_min[random_idx], X_seed_min[random_neighbor_idx]))
        
        return np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(self.minority_label, len(samples))])
        
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'min_precision': self.min_precision, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

class SMOTE_RSB(OverSampling):
    """
        Notes:
            * I think the description of the algorithm in Fig 5 of the paper is not correct. The set "resultSet" is initialized with the original instances, and then the While loop in the Algorithm run until resultSet is empty, which never holds. Also, the resultSet is only extended in the loop. Our implementation is changed in the following way: we generate twice as many instances are required to balance the dataset, and repeat the loop until the number of new samples added to the training set is enough to balance the dataset.
    """
    
    categories= [OverSampling.cat_extensive,
                OverSampling.cat_sample_ordinary]
    
    def __init__(self, proportion= 2.0, n_neighbors= 5, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): number of neighbors in the SMOTE sampling
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7]})
    
    def sample(self, X, y,weight,ntree):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        self.class_label_statistics(X, y)
        
        if self.class_stats[self.minority_label] < 2:
            return X.copy(), y.copy()
        
        X_maj= X[y == self.majority_label]
        X_min= X[y == self.minority_label]
        
        # Step 1: do the sampling
        smote= SMOTE(proportion= self.proportion, n_neighbors= self.n_neighbors, n_jobs= self.n_jobs, random_state= self.random_state)
        X_samp, y_samp= smote.sample(X, y,weight,ntree)
        X_samp, y_samp= X_samp[len(X):], y_samp[len(X):]
        
        if len(X_samp) == 0: return X.copy(), y.copy()
        
        # Step 2: (original will be added later)
        result_set= []
        
        # Step 3: first the data is normalized
        maximums= np.max(X_samp, axis= 0)
        minimums= np.min(X_samp, axis= 0)
        
        # normalize X_new and X_maj
        norm_factor= maximums - minimums
        norm_factor[norm_factor == 0]= np.max(np.vstack([maximums[norm_factor == 0], np.repeat(1, np.sum(norm_factor == 0))]), axis= 0)
        X_samp_norm= X_samp / norm_factor
        X_maj_norm= X_maj / norm_factor
        
        # compute similarity matrix
        similarity_matrix= 1.0 - pairwise_distances(X_samp_norm, X_maj_norm, metric= 'minkowski', p= 1)/len(X[0])
        
        # Step 4: counting the similar examples
        similarity_value= 0.4
        syn= len(X_samp)
        cont= np.zeros(syn)
        
        already_added= np.repeat(False, len(X_samp))
        
        while len(result_set) < len(X_maj) - len(X_min) and similarity_value <= 0.9:
            for i in range(syn):
                cont[i]= np.sum(similarity_matrix[i,:] > similarity_value)
                if cont[i] == 0 and not already_added[i]:
                    result_set.append(X_samp[i])
                    already_added[i]= True
            similarity_value= similarity_value + 0.05
        
        # Step 5: returning the results depending the number of instances added to the result set
        if len(result_set) > 0:
            return np.vstack([X, np.vstack(result_set)]), np.hstack([y, np.repeat(self.minority_label, len(result_set))])
        else:
            return np.vstack([X, X_samp]), np.hstack([y, y_samp])
        
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class NRSBoundary_SMOTE(OverSampling):
    """
    References:
        * BibTex::
            
            @Article{nrsboundary_smote,
                    author= {Feng, Hu and Hang, Li},
                    title= {A Novel Boundary Oversampling Algorithm Based on Neighborhood Rough Set Model: NRSBoundary-SMOTE},
                    journal= {Mathematical Problems in Engineering},
                    year= {2013},
                    pages= {10},
                    doi= {10.1155/2013/694809},
                    url= {http://dx.doi.org/10.1155/694809}
                    }
    """
    
    categories= [OverSampling.cat_extensive,
                 OverSampling.cat_borderline]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, w= 0.005, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbors component
            w (float): used to set neighborhood radius
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(w, "w", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.w= w
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7], 
                                                    'w': [0.005, 0.01, 0.05]})
    
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        if self.class_stats[self.minority_label] < 2:
            # _logger.warning(self.__class__.__name__ + ": " + "The number of minority samples (%d) is not enough for sampling" % self.class_stats[self.minority_label])
            return X.copy(), y.copy()
        
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        # step 1
        bound_set= []
        pos_set= []
        
        # step 2
        X_min_indices= np.where(y == self.minority_label)[0]
        X_min= X[X_min_indices]
        
        # step 3
        dm= pairwise_distances(X, X)
        d_max= np.max(dm, axis= 1)
        max_dist= np.max(dm)
        np.fill_diagonal(dm, max_dist)
        d_min= np.min(dm, axis= 1)
        
        delta= d_min + self.w*(d_max - d_min)
        
        # number of neighbors is not interesting here, as we use the
        # radius_neighbors function to extract the neighbors in a given radius
        nn= NearestNeighbors(n_neighbors= self.n_neighbors + 1, n_jobs= self.n_jobs)
        nn.fit(X)
        for i in range(len(X)):
            indices= nn.radius_neighbors(X[i].reshape(1, -1), delta[i], return_distance= False)
            if y[i] == self.minority_label and not np.sum(y[indices[0]] == self.minority_label) == len(indices[0]):
                bound_set.append(i)
            elif y[i] == self.majority_label and np.sum(y[indices[0]] == self.majority_label) == len(indices[0]):
                pos_set.append(i)
        
        bound_set= np.array(bound_set)
        pos_set= np.array(pos_set)
        
        if len(pos_set) == 0 or len(bound_set) == 0:
            return X.copy(), y.copy()
        
        # step 4 and 5
        # computing the nearest neighbors of the bound set from the minority set
        nn= NearestNeighbors(n_neighbors= min([len(X_min), self.n_neighbors + 1]), n_jobs= self.n_jobs)
        nn.fit(X_min)
        distances, indices= nn.kneighbors(X[bound_set])
        
        # do the sampling
        samples= []
        trials= 0
        w= self.w
        while len(samples) < num_to_sample:
            idx= self.random_state.choice(len(bound_set))
            random_neighbor_idx= self.random_state.choice(indices[idx][1:])
            x_new= self.sample_between_points(X[bound_set[idx]], X_min[random_neighbor_idx])
            
            # checking the conflict
            dist_from_pos_set= np.linalg.norm(X[pos_set] - x_new, axis= 1)
            if np.all(dist_from_pos_set > delta[pos_set]):
                # no conflict
                samples.append(x_new)
            trials= trials + 1
            if trials > 1000 and len(samples) == 0:
                trials= 0
                w= w*0.9
            
        return np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(self.minority_label, len(samples))])
        
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'w': self.w, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}



class LN_SMOTE(OverSampling):
    """
    References:
        * BibTex::
            
            @INPROCEEDINGS{ln_smote, 
                            author={Maciejewski, T. and Stefanowski, J.}, 
                            booktitle={2011 IEEE Symposium on Computational Intelligence and Data Mining (CIDM)}, 
                            title={Local neighbourhood extension of SMOTE for mining imbalanced data}, 
                            year={2011}, 
                            volume={}, 
                            number={}, 
                            pages={104-111}, 
                            keywords={Bayes methods;data mining;pattern classification;local neighbourhood extension;imbalanced data mining;focused resampling technique;SMOTE over-sampling method;naive Bayes classifiers;Noise measurement;Noise;Decision trees;Breast cancer;Sensitivity;Data mining;Training}, 
                            doi={10.1109/CIDM.2011.5949434}, 
                            ISSN={}, 
                            month={April}}
    """
    
    categories= [OverSampling.cat_extensive,
                 OverSampling.cat_sample_componentwise]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): parameter of the NearestNeighbors component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7]})
    
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        # number of samples to generate
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        if self.n_neighbors + 2 > len(X):
            n_neighbors= len(X) - 2
        else:
            n_neighbors= self.n_neighbors
        
        if n_neighbors < 2:
            return X.copy(), y.copy()
        
        # nearest neighbors of each instance to each instance in the dataset
        nn= NearestNeighbors(n_neighbors= n_neighbors + 2, n_jobs= self.n_jobs)
        nn.fit(X)
        distances, indices= nn.kneighbors(X)
        
        minority_indices= np.where(y == self.minority_label)[0]
        
        # dimensionality
        d= len(X[0])
        
        def safe_level(p_idx, n_idx= None):
            """
            computing the safe level of samples
            
            Args:
                p_idx (int): index of positive sample
                n_idx (int): index of other sample
                
            Returns:
                int: safe level
            """
            if n_idx is None:
                # implementation for 1 sample only
                return np.sum(y[indices[p_idx][1:-1]] == self.minority_label)
            else:
                # implementation for 2 samples
                if (not y[n_idx] != self.majority_label) and p_idx in indices[n_idx][1:-1]:
                    # -1 because p_idx will be replaced
                    n_positives= np.sum(y[indices[n_idx][1:-1]] == self.minority_label) - 1
                    if y[indices[n_idx][-1]] == self.minority_label:
                        # this is the effect of replacing p_idx by the next (k+1)th neighbor
                        n_positives= n_positives + 1
                    return n_positives
                return np.sum(y[indices[n_idx][1:-1]] == self.minority_label)
        
        def random_gap(slp, sln, n_label):
            """
            determining random gap
            
            Args:
                slp (int): safe level of p
                sln (int): safe level of n
                n_label (int): label of n
                
            Returns:
                float: gap
            """
            delta= 0
            if sln == 0 and slp > 0:
                return delta
            else:
                sl_ratio= slp/sln
                if sl_ratio == 1:
                    delta= self.random_state.random_sample()
                elif sl_ratio > 1:
                    delta= self.random_state.random_sample()/sl_ratio
                else:
                    delta= 1.0 - self.random_state.random_sample()*sl_ratio
            if not n_label == self.minority_label:
                delta= delta*sln/(n_neighbors)
            return delta
        
        # generating samples
        trials= 0
        samples= []
        while len(samples) < num_to_sample:
            p_idx= self.random_state.choice(minority_indices)
            # extract random neighbor of p
            n_idx= self.random_state.choice(indices[p_idx][1:-1])
            
            # checking can-create criteria
            slp= safe_level(p_idx)
            sln= safe_level(p_idx, n_idx)
            
            if (not slp == 0) or (not sln == 0):
                # can create
                p= X[p_idx]
                n= X[n_idx]
                x_new= p.copy()
                
                for a in range(d):
                    delta= random_gap(slp, sln, y[n_idx])
                    diff= n[a] - p[a]
                    x_new[a]= p[a] + delta*diff
                samples.append(x_new)
            
            trials= trials + 1
            if len(samples)/trials < 1.0/num_to_sample:
                # _logger.info(self.__class__.__name__ + ": " + "no instances with slp > 0 and sln > 0 found")
                return X.copy(), y.copy()
        
        return np.vstack([X, samples]), np.hstack([y, np.repeat(self.minority_label, len(samples))])
        
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

class MWMOTE(OverSampling):
    """
    Notes:
        * The original method was not prepared for the case of having clusters of 1 elements.
    """
    
    categories= [OverSampling.cat_extensive,
                 OverSampling.cat_uses_clustering,
                 OverSampling.cat_borderline]
    
    def __init__(self, proportion= 1.0, k1= 5, k2= 5, k3= 5, M= 10, cf_th= 5.0, cmax= 10.0, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            k1 (int): parameter of the NearestNeighbors component
            k2 (int): parameter of the NearestNeighbors component
            k3 (int): parameter of the NearestNeighbors component
            M (int): number of clusters
            cf_th (float): cutoff threshold
            cmax (float): maximum closeness value
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(k1, 'k1', 1)
        self.check_greater_or_equal(k2, 'k2', 1)
        self.check_greater_or_equal(k3, 'k3', 1)
        self.check_greater_or_equal(M, 'M', 1)
        self.check_greater_or_equal(cf_th, 'cf_th', 0)
        self.check_greater_or_equal(cmax, 'cmax', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.k1= k1
        self.k2= k2
        self.k3= k3
        self.M= M
        self.cf_th= cf_th
        self.cmax= cmax
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'k1': [5, 9], 
                                                    'k2': [5, 9], 
                                                    'k3': [5, 9], 
                                                    'M': [4, 10], 
                                                    'cf_th': [5.0], 
                                                    'cmax': [10.0]})
    
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        X_min= X[y == self.minority_label]
        X_maj= X[y == self.majority_label]
        
        minority= np.where(y == self.minority_label)[0]
        
        # Step 1
        nn= NearestNeighbors(n_neighbors= min([len(X), self.k1 + 1]), n_jobs= self.n_jobs)
        nn.fit(X)
        dist1, ind1= nn.kneighbors(X)
        
        # Step 2
        filtered_minority= np.array([i for i in minority if np.sum(y[ind1[i][1:]] == self.minority_label) > 0])
        if len(filtered_minority) == 0:
            # _logger.info(self.__class__.__name__ + ": " + "filtered_minority array is empty")
            return X.copy(), y.copy()
        
        # Step 3 - ind2 needs to be indexed by indices of the lengh of X_maj
        nn_maj= NearestNeighbors(n_neighbors= self.k2, n_jobs= self.n_jobs)
        nn_maj.fit(X_maj)
        dist2, ind2= nn_maj.kneighbors(X[filtered_minority])
        
        # Step 4
        border_majority= np.unique(ind2.flatten())
        
        # Step 5 - ind3 needs to be indexed by indices of the length of X_min
        nn_min= NearestNeighbors(n_neighbors= min([self.k3, len(X_min)]), n_jobs= self.n_jobs)
        nn_min.fit(X_min)
        dist3, ind3= nn_min.kneighbors(X_maj[border_majority])
        
        # Step 6 - informative minority indexes X_min
        informative_minority= np.unique(ind3.flatten())
        
        def closeness_factor(y, x, cf_th= self.cf_th, cmax= self.cmax):
            """
            Closeness factor according to the Eq (6)
            
            Args:
                y (np.array): training instance (border_majority)
                x (np.array): training instance (informative_minority)
                cf_th (float): cutoff threshold
                cmax (float): maximum values
                
            Returns:
                float: closeness factor
            """
            d= np.linalg.norm(y - x)/len(y)
            if d == 0.0:
                d= 0.1
            if 1.0/d < cf_th:
                f= 1.0/d
            else:
                f= cf_th
            return f/cf_th*cmax
        
        # Steps 7 - 9
        # _logger.info(self.__class__.__name__ + ": " +'computing closeness factors')        
        closeness_factors= np.zeros(shape=(len(border_majority), len(informative_minority)))
        for i in range(len(border_majority)):
            for j in range(len(informative_minority)):
                closeness_factors[i,j]= closeness_factor(X_maj[border_majority[i]], X_min[informative_minority[j]])
        
        # _logger.info(self.__class__.__name__ + ": " +'computing information weights')
        information_weights= np.zeros(shape=(len(border_majority), len(informative_minority)))
        for i in range(len(border_majority)):
            norm_factor= np.sum(closeness_factors[i,:])
            for j in range(len(informative_minority)):
                information_weights[i,j]= closeness_factors[i,j]**2/norm_factor
        
        selection_weights= np.sum(information_weights, axis= 0)
        selection_probabilities= selection_weights/np.sum(selection_weights)
        
        # Step 10
        # _logger.info(self.__class__.__name__ + ": " +'do clustering')
        kmeans= KMeans(n_clusters= min([len(X_min), self.M]), n_jobs= self.n_jobs, random_state= self.random_state)
        kmeans.fit(X_min)
        imin_labels= kmeans.labels_[informative_minority]
        
        clusters= [np.where(imin_labels == i)[0] for i in range(np.max(kmeans.labels_)+1)]
        
        # Step 11
        samples= []
        
        # Step 12
        for i in range(num_to_sample):
            random_index= self.random_state.choice(informative_minority, p= selection_probabilities)
            cluster_label= kmeans.labels_[random_index]
            random_index_in_cluster= self.random_state.choice(clusters[cluster_label])
            samples.append(self.sample_between_points(X_min[random_index], X_min[random_index_in_cluster]))
        
        return np.vstack([X, samples]), np.hstack([y, np.repeat(self.minority_label, len(samples))])
        
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'k1': self.k1, 
                'k2': self.k2, 
                'k3': self.k3, 
                'M': self.M, 
                'cf_th': self.cf_th, 
                'cmax': self.cmax, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SMOTE_IPF(OverSampling):
    
    categories= [OverSampling.cat_changes_majority,
                 OverSampling.cat_uses_classifier]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, n_folds= 9, k= 3, p= 0.01, voting= 'majority', classifier= DecisionTreeClassifier(random_state= 2), n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): number of neighbors in SMOTE sampling
            n_folds (int): the number of partitions
            k (int): used in stopping condition
            p (float): percentage value ([0,1]) used in stopping condition
            voting (str): 'majority'/'consensus'
            classifier (obj): classifier object
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_folds, "n_folds", 2)
        self.check_greater_or_equal(k, "k", 1)
        self.check_greater_or_equal(p, "p", 0)
        self.check_isin(voting, "voting", ['majority', 'consensus'])
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.n_folds= n_folds
        self.k= k
        self.p= p
        self.voting= voting
        self.classifier= classifier
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7], 
                                                    'n_folds': [9], 
                                                    'k': [3], 
                                                    'p': [0.01], 
                                                    'voting': ['majority', 'consensus'], 
                                                    'classifier': [DecisionTreeClassifier(random_state= 2)]})
    
    def sample(self, X, y,weight,ntree):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        self.class_label_statistics(X, y)
        
        if self.class_stats[self.minority_label] < 2:
            return X.copy(), y.copy()
        
        # do SMOTE sampling
        X_samp, y_samp= SMOTE(self.proportion, self.n_neighbors, n_jobs= self.n_jobs, random_state= self.random_state).sample(X, y,weight,ntree)
        n_folds= min([self.n_folds, np.sum(y == self.minority_label)])
        
        condition= 0
        while True:
            # validating the sampled dataset
            validator= StratifiedKFold(n_folds)
            predictions= []
            for train_index, _ in validator.split(X_samp, y_samp):
                self.classifier.fit(X_samp[train_index], y_samp[train_index])
                predictions.append(self.classifier.predict(X_samp))
            
            # do decision based on one of the voting schemes
            if self.voting == 'majority':
                pred_votes= (np.mean(predictions, axis= 0) > 0.5).astype(int)
                to_remove= np.where(np.not_equal(pred_votes, y_samp))[0]
            elif self.voting == 'consensus':
                pred_votes= (np.mean(predictions, axis= 0) > 0.5).astype(int)
                sum_votes= np.sum(predictions, axis= 0)
                to_remove= np.where(np.logical_and(np.not_equal(pred_votes, y_samp), np.equal(sum_votes, self.n_folds)))[0]
            else:
                raise ValueError(self.__class__.__name__ + ": " + 'Voting scheme %s is not implemented' % self.voting)
            
            # delete samples incorrectly classified
            X_samp= np.delete(X_samp, to_remove, axis= 0)
            y_samp= np.delete(y_samp, to_remove)
            
            # if the number of samples removed becomes small or k iterations were done quit
            if len(to_remove) < len(X_samp)*self.p:condition= condition + 1
            else:condition= 0
            if condition >= self.k:break
        return X_samp, y_samp
        
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'n_folds': self.n_folds, 
                'k': self.k, 
                'p': self.p, 
                'voting': self.voting, 
                'n_jobs': self.n_jobs, 
                'classifier': self.classifier,
                'random_state': self._random_state_init}


class SMOTE_PSO(OverSampling):
    """

        Notes:
            * I find the description of the technique a bit confusing, especially on the bounds of the search space of velocities and positions. Equations 15 and 16 specify the lower and upper bounds, the lower bound is in fact a vector while the upper bound is a distance. I tried to implement something meaningful.
            * I also find the setting of accelerating constant 2.0 strange, most of the time the velocity will be bounded due to this choice. 
            * Also, training and predicting probabilities with a non-linear SVM as the evaluation function becomes fairly expensive when the number of training vectors reaches a couple of thousands. To reduce computational burden, minority and majority vectors far from the other class are removed to reduce the size of both classes to a maximum of 500 samples. Generally, this shouldn't really affect the results as the technique focuses on the samples near the class boundaries.
    """
    
    categories= [OverSampling.cat_extensive,
                 OverSampling.cat_memetic,
                 OverSampling.cat_uses_classifier]
    
    def __init__(self, k= 3, eps= 0.05, n_pop= 10, w= 1.0, c1= 2.0, c2= 2.0, num_it= 10, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            k (int): number of neighbors in nearest neighbors component, this is also the
                        multiplication factor of minority support vectors
            eps (float): use to specify the initially generated support vectors along minority-
                            majority lines
            n_pop (int): size of population
            w (float): intertia constant
            c1 (float): acceleration constant of local optimum
            c2 (float): acceleration constant of population optimum
            num_it (int): number of iterations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(k, "k", 1)
        self.check_greater(eps, "eps", 0)
        self.check_greater_or_equal(n_pop, "n_pop", 1)
        self.check_greater_or_equal(w, "w", 0)
        self.check_greater_or_equal(c1, "c1", 0)
        self.check_greater_or_equal(c2, "c2", 0)
        self.check_greater_or_equal(num_it, "num_it", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.k= k
        self.eps= eps
        self.n_pop= n_pop
        self.w= w
        self.c1= c1
        self.c2= c2
        self.num_it= num_it
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'k': [3, 5, 7], 
                                                    'eps': [0.05], 
                                                    'n_pop': [5], 
                                                    'w': [0.5, 1.0], 
                                                    'c1': [1.0, 2.0], 
                                                    'c2': [1.0, 2.0], 
                                                    'num_it': [5]})
    
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        # saving original dataset
        X_orig= X
        y_orig= y
        
        # scaling the records
        mms= MinMaxScaler()
        X_scaled= mms.fit_transform(X)
        
        # removing majority and minority samples far from the training data if needed to
        # increase performance
        performance_threshold= 500
        
        n_maj_to_remove= np.sum(y == self.majority_label) - performance_threshold
        if n_maj_to_remove > 0:
            # if majority samples are to be removed
            nn= NearestNeighbors(n_neighbors= 1, n_jobs= self.n_jobs).fit(X_scaled[y == self.minority_label])
            dist, ind= nn.kneighbors(X_scaled)
            di= sorted([(dist[i][0], i) for i in range(len(ind))], key= lambda x: x[0])
            to_remove= []
            # finding the proper number of samples farest from the minority samples
            for i in reversed(range(len(di))):
                if y[di[i][1]] == self.majority_label:
                    to_remove.append(di[i][1])
                if len(to_remove) >= n_maj_to_remove:
                    break
            # removing the samples
            X_scaled= np.delete(X_scaled, to_remove, axis= 0)
            y= np.delete(y, to_remove)
            
        n_min_to_remove= np.sum(y == self.minority_label) - performance_threshold
        if n_min_to_remove > 0:
            # if majority samples are to be removed
            nn= NearestNeighbors(n_neighbors= 1, n_jobs= self.n_jobs).fit(X_scaled[y == self.majority_label])
            dist, ind= nn.kneighbors(X_scaled)
            di= sorted([(dist[i][0], i) for i in range(len(ind))], key= lambda x: x[0])
            to_remove= []
            # finding the proper number of samples farest from the minority samples
            for i in reversed(range(len(di))):
                if y[di[i][1]] == self.minority_label:
                    to_remove.append(di[i][1])
                if len(to_remove) >= n_min_to_remove:
                    break
            # removing the samples
            X_scaled= np.delete(X_scaled, to_remove, axis= 0)
            y= np.delete(y, to_remove)
        
        # fitting SVM to extract initial support vectors
        svc= SVC(kernel= 'rbf', probability= True, gamma= 'auto', random_state= self.random_state)
        svc.fit(X_scaled, y)
        
        # extracting the support vectors
        SV_min= np.array([i for i in svc.support_ if y[i] == self.minority_label])
        SV_maj= np.array([i for i in svc.support_ if y[i] == self.majority_label])
        
        X_SV_min= X_scaled[SV_min]
        X_SV_maj= X_scaled[SV_maj]
        
        # finding nearest majority support vectors
        nn= NearestNeighbors(n_neighbors= min([len(X_SV_maj), self.k]), n_jobs= self.n_jobs)
        nn.fit(X_SV_maj)
        dist, ind= nn.kneighbors(X_SV_min)
        
        # finding the initial particle and specifying the search space
        X_min_gen= []
        search_space= []
        init_velocity= []
        for i in range(len(SV_min)):
            for j in range(min([len(X_SV_maj), self.k])):
                min_vector= X_SV_min[i]
                maj_vector= X_SV_maj[ind[i][j]]
                # the upper bound of the search space if specified by the closest majority support vector
                upper_bound= X_SV_maj[ind[i][0]]
                # the third element of the search space specification is the distance of the vector and the closest
                # majority support vector, which specifies the radius of the search
                search_space.append([min_vector, maj_vector, np.linalg.norm(min_vector - upper_bound)])
                # initial particles
                X_min_gen.append(min_vector + self.eps*(maj_vector - min_vector))
                # initial velocities
                init_velocity.append(self.eps*(maj_vector - min_vector))
        
        X_min_gen= np.vstack(X_min_gen)
        init_velocity= np.vstack(init_velocity)
        
        # evaluates a specific particle
        def evaluate(X_train, y_train, X_test, y_test):
            """
            Trains support vector classifier and evaluates it
            
            Args:
                X_train (np.matrix): training vectors
                y_train (np.array): target labels
                X_test (np.matrix): test vectors
                y_test (np.array): test labels
            """
            svc.fit(X_train, y_train)
            y_pred= svc.predict_proba(X_test)[:,np.where(svc.classes_ == self.minority_label)[0][0]]
            return roc_auc_score(y_test, y_pred)
        
        # initializing the particle swarm and the particle and population level
        # memory
        particle_swarm= [X_min_gen.copy() for _ in range(self.n_pop)]
        velocities= [init_velocity.copy() for _ in range(self.n_pop)]
        local_best= [X_min_gen.copy() for _ in range(self.n_pop)]
        local_best_scores= [0.0]*self.n_pop
        global_best= X_min_gen.copy()
        global_best_score= 0.0

        for i in range(self.num_it):
            # _logger.info(self.__class__.__name__ + ": " +"Iteration %d" % i)
            # evaluate population
            scores= [evaluate(np.vstack([X_scaled, p]), np.hstack([y, np.repeat(self.minority_label, len(p))]), X_scaled, y) for p in particle_swarm]
            
            # update best scores
            for i, s in enumerate(scores):
                if s > local_best_scores[i]:
                    local_best_scores[i]= s
                    local_best[i]= particle_swarm[i]
                if s > global_best_score:
                    global_best_score= s
                    global_best= particle_swarm[i]
            
            # update velocities
            for i, p in enumerate(particle_swarm):
                velocities[i]= self.w*velocities[i] + self.c1*self.random_state.random_sample()*(local_best[i] - p) + self.c2*self.random_state.random_sample()*(global_best - p)
            
            # bound velocities according to search space constraints
            for v in velocities:
                for i in range(len(v)):
                    if np.linalg.norm(v[i]) > search_space[i][2]/2.0:
                        v[i]= v[i]/np.linalg.norm(v[i])*search_space[i][2]/2.0
            
            # update positions
            for i, p in enumerate(particle_swarm):
                particle_swarm[i]= particle_swarm[i] + velocities[i]
            
            # bound positions according to search space constraints
            for p in particle_swarm:
                for i in range(len(p)):
                    if np.linalg.norm(p[i] - search_space[i][0]) > search_space[i][2]:
                        p[i]= search_space[i][0] + (p[i] - search_space[i][0])/np.linalg.norm(p[i] - search_space[i][0])*search_space[i][2]
            
        return np.vstack([X_orig, mms.inverse_transform(global_best)]), np.hstack([y_orig, np.repeat(self.minority_label, len(global_best))])
        
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'k': self.k, 
                'eps': self.eps, 
                'n_pop': self.n_pop, 
                'w': self.w, 
                'c1': self.c1, 
                'c2': self.c2, 
                'num_it': self.num_it, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

class Edge_Det_SMOTE(OverSampling):
    """
    
    
    Notes:
        * This technique is very loosely specified.
    """
    
    categories= [OverSampling.cat_density_based,
                 OverSampling.cat_borderline,
                 OverSampling.cat_extensive]
    
    def __init__(self, proportion= 1.0, k= 5, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            k (int): number of neighbors
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(k, "k", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.k= k
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'k': [3, 5, 7]})
    
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        if self.class_stats[self.minority_label] < 2:
            # _logger.warning(self.__class__.__name__ + ": " + "The number of minority samples (%d) is not enough for sampling" % self.class_stats[self.minority_label])
            return X.copy(), y.copy()
        
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        d= len(X[0])
        X_min= X[y == self.minority_label]
        
        # organizing class labels according to feature ranking
        magnitudes= np.zeros(len(X))
        for i in range(d):
            _, idx, label= zip(*sorted(zip(X[:,i], np.arange(len(X)), y), key= lambda x: x[0]))
            # extracting edge magnitudes in this dimension
            for j in range(1, len(idx)-1):
                magnitudes[idx[j]]= magnitudes[idx[j]] + (label[j-1] - label[j+1])**2
        
        # density estimation
        magnitudes= magnitudes[y == self.minority_label]
        magnitudes= np.sqrt(magnitudes)
        magnitudes= magnitudes/np.sum(magnitudes)
        
        # fitting nearest neighbors models to minority samples
        nn= NearestNeighbors(n_neighbors= min([len(X_min), self.k+1]), n_jobs= self.n_jobs)
        nn.fit(X_min)
        dist, ind= nn.kneighbors(X_min)
        
        # do the sampling
        samples= []
        for _ in range(num_to_sample):
            idx= self.random_state.choice(np.arange(len(X_min)), p= magnitudes)
            samples.append(self.sample_between_points(X_min[idx], X_min[self.random_state.choice(ind[idx][1:])]))
        
        return np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(self.minority_label, len(samples))])
    
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'k': self.k, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class kmeans_SMOTE(OverSampling):
   
    
    categories= [OverSampling.cat_extensive,
                 OverSampling.cat_uses_clustering]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, n_clusters= 10, irt= 2.0, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): number of neighbors
            n_clusters (int): number of clusters
            irt (float): imbalanced ratio threshold
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clusters, "n_clusters", 1)
        self.check_greater_or_equal(irt, "irt", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.n_clusters= n_clusters
        self.irt= irt
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7], 
                                                    'n_clusters': [2, 5, 10, 20, 50], 
                                                    'irt': [0.5, 0.8, 1.0, 1.5]})
    
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        # applying kmeans clustering to all data
        n_clusters= min([self.n_clusters, len(X)])
        kmeans= KMeans(n_clusters= n_clusters, n_jobs= self.n_jobs, random_state= self.random_state)
        kmeans.fit(X)
        
        # extracting clusters
        labels= kmeans.labels_
        clusters= [np.where(labels == l)[0] for l in range(n_clusters)]
        
        # cluster filtering
        filt_clusters= [c for c in clusters if (np.sum(y[c] == self.majority_label) + 1)/(np.sum(y[c] == self.minority_label) + 1) < self.irt and np.sum(y[c] == self.minority_label) > 1]
        
        if len(filt_clusters) == 0:
            # _logger.warning(self.__class__.__name__ + ": " +"number of clusters after filtering is 0")
            return X.copy(), y.copy()
        
        # Step 2 in the paper
        sparsity= []
        nearest_neighbors= []
        cluster_minority_ind= []
        for c in filt_clusters:
            # extract minority indices in the cluster
            minority_ind= c[y[c] == self.minority_label]
            cluster_minority_ind.append(minority_ind)
            # compute distance matrix of minority samples in the cluster
            dm= pairwise_distances(X[minority_ind])
            min_count= len(minority_ind)
            # compute the average of distances
            avg_min_dist= (np.sum(dm) - dm.trace())/(len(minority_ind)**2 - len(minority_ind))
            # compute sparsity (Step 4)
            sparsity.append(avg_min_dist**len(X[0])/min_count)
            # extract the nearest neighbors graph
            nearest_neighbors.append(NearestNeighbors(n_neighbors= min([len(minority_ind), self.n_neighbors + 1]), n_jobs= self.n_jobs).fit(X[minority_ind]).kneighbors(X[minority_ind]))
        
        # Step 5 - compute density of sampling
        weights= sparsity/np.sum(sparsity)
        
        # do the sampling
        samples= []
        while len(samples) < num_to_sample:
            # choose random cluster index and random minority element
            clust_ind= self.random_state.choice(np.arange(len(weights)), p= weights)
            idx= self.random_state.randint(len(cluster_minority_ind[clust_ind]))
            base_idx= cluster_minority_ind[clust_ind][idx]
            # choose random neighbor
            neighbor_idx= self.random_state.choice(cluster_minority_ind[clust_ind][nearest_neighbors[clust_ind][1][idx][1:]])
            # sample
            samples.append(self.sample_between_points(X[base_idx], X[neighbor_idx]))
            
        return np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(self.minority_label, len(samples))])

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'n_clusters': self.n_clusters, 
                'irt': self.irt, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SN_SMOTE(OverSampling):
    """
    References:
        * BibTex::
            
            @Article{sn_smote,
                        author="Garc{\'i}a, V.
                        and S{\'a}nchez, J. S.
                        and Mart{\'i}n-F{\'e}lez, R.
                        and Mollineda, R. A.",
                        title="Surrounding neighborhood-based SMOTE for learning from imbalanced data sets",
                        journal="Progress in Artificial Intelligence",
                        year="2012",
                        month="Dec",
                        day="01",
                        volume="1",
                        number="4",
                        pages="347--362",
                        abstract="Many traditional approaches to pattern classification assume that the problem classes share similar prior probabilities. However, in many real-life applications, this assumption is grossly violated. Often, the ratios of prior probabilities between classes are extremely skewed. This situation is known as the class imbalance problem. One of the strategies to tackle this problem consists of balancing the classes by resampling the original data set. The SMOTE algorithm is probably the most popular technique to increase the size of the minority class by generating synthetic instances. From the idea of the original SMOTE, we here propose the use of three approaches to surrounding neighborhood with the aim of generating artificial minority instances, but taking into account both the proximity and the spatial distribution of the examples. Experiments over a large collection of databases and using three different classifiers demonstrate that the new surrounding neighborhood-based SMOTE procedures significantly outperform other existing over-sampling algorithms.",
                        issn="2192-6360",
                        doi="10.1007/s13748-012-0027-5",
                        url="https://doi.org/10.1007/s13748-012-0027-5"
                        }
    """
    
    categories= [OverSampling.cat_extensive,
                 OverSampling.cat_sample_ordinary]
    
    def __init__(self, proportion= 1.0, n_neighbors= 5, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (float): number of neighbors
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7]})
    
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        if self.class_stats[self.minority_label] < 2:
            # _logger.warning(self.__class__.__name__ + ": " + "The number of minority samples (%d) is not enough for sampling" % self.class_stats[self.minority_label])
            return X.copy(), y.copy()
        
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        X_min= X[y == self.minority_label]
        
        # the search for the k nearest centroid neighbors is limited for the nearest
        # 10*n_neighbors neighbors
        nn= NearestNeighbors(n_neighbors= min([self.n_neighbors*10, len(X_min)]), n_jobs= self.n_jobs)
        nn.fit(X_min)
        dist, ind= nn.kneighbors(X_min)
        
        # determining k nearest centroid neighbors
        ncn= np.zeros(shape=(len(X_min), self.n_neighbors)).astype(int)
        ncn_nums= np.zeros(len(X_min)).astype(int)
        
        # extracting nearest centroid neighbors
        for i in range(len(X_min)):
            # the first NCN neighbor is the first neighbor
            ncn[i, 0]= ind[i][1]
            
            # iterating through all neighbors and finding the one with smaller
            # centroid distance to X_min[i] than the previous set of neighbors
            n_cent= 1
            centroid= X_min[ncn[i, 0]]
            cent_dist= np.linalg.norm(centroid - X_min[i])
            j= 2
            while j < len(ind[i]) and n_cent < self.n_neighbors:
                new_cent_dist= np.linalg.norm((centroid + X_min[ind[i][j]])/(n_cent + 1) - X_min[i])
                
                # checking if new nearest centroid neighbor found
                if new_cent_dist < cent_dist:
                    centroid= centroid + X_min[ind[i][j]]
                    ncn[i, n_cent]= ind[i][j]
                    n_cent= n_cent + 1
                    cent_dist= new_cent_dist
                j= j + 1
            
            # registering the number of nearest centroid neighbors found
            ncn_nums[i]= n_cent
        
        # generating samples
        samples= []
        while len(samples) < num_to_sample:
            random_idx= self.random_state.randint(len(X_min))
            random_neighbor_idx= self.random_state.choice(ncn[random_idx][:ncn_nums[random_idx]])
            samples.append(self.sample_between_points(X_min[random_idx], X_min[random_neighbor_idx]))
            
        return np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(self.minority_label, len(samples))])

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

class cluster_SMOTE(OverSampling):
    """
    References:
        * BibTex::
            
            @INPROCEEDINGS{cluster_SMOTE, 
                            author={Cieslak, D. A. and Chawla, N. V. and Striegel, A.}, 
                            booktitle={2006 IEEE International Conference on Granular Computing}, 
                            title={Combating imbalance in network intrusion datasets}, 
                            year={2006}, 
                            volume={}, 
                            number={}, 
                            pages={732-737}, 
                            keywords={Intelligent networks;Intrusion detection;Telecommunication traffic;Data mining;Computer networks;Data security;Machine learning;Counting circuits;Computer security;Humans}, 
                            doi={10.1109/GRC.2006.1635905}, 
                            ISSN={}, 
                            month={May}}
    """
    
    categories= [OverSampling.cat_extensive,
                 OverSampling.cat_uses_clustering]
    
    def __init__(self, proportion= 1.0, n_neighbors= 3, n_clusters= 3, n_jobs= 1, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            proportion (float): proportion of the difference of n_maj and n_min to sample
                                    e.g. 1.0 means that after sampling the number of minority
                                    samples will be equal to the number of majority samples
            n_neighbors (int): number of neighbors in SMOTE
            n_clusters (int): number of clusters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state, like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clusters, "n_components", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')
        
        self.proportion= proportion
        self.n_neighbors= n_neighbors
        self.n_clusters= n_clusters
        self.n_jobs= n_jobs
        
        self.set_random_state(random_state)
        
    @classmethod
    def parameter_combinations(cls):
        """
        Generates reasonable paramter combinations.
        
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'proportion': [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                                                    'n_neighbors': [3, 5, 7], 
                                                    'n_clusters': [3, 5, 7, 9]})
    
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +"Running sampling via %s" % self.descriptor())
        
        self.class_label_statistics(X, y)
        
        X_min= X[y == self.minority_label]
        
        # determining the number of samples to generate
        num_to_sample= self.number_of_instances_to_sample(self.proportion, self.class_stats[self.majority_label], self.class_stats[self.minority_label])
        
        if num_to_sample == 0:
            # _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        kmeans= KMeans(n_clusters= min([len(X_min), self.n_clusters]), n_jobs= self.n_jobs, random_state= self.random_state)
        kmeans.fit(X_min)
        cluster_labels= kmeans.labels_
        unique_labels= np.unique(cluster_labels)
        
        # creating nearest neighbors objects for each cluster
        cluster_indices= [np.where(cluster_labels == c)[0] for c in unique_labels]
        cluster_nns= [NearestNeighbors(n_neighbors= min([self.n_neighbors, len(cluster_indices[idx])])).fit(X_min[cluster_indices[idx]]) for idx in range(len(cluster_indices))]
        
        if max([len(c) for c in cluster_indices]) <= 1:
            # _logger.info(self.__class__.__name__ + ": " + "All clusters contain 1 element")
            return X.copy(), y.copy()
        
        # generating the samples
        samples= []
        while len(samples) < num_to_sample:
            cluster_idx= self.random_state.randint(len(cluster_indices))
            if len(cluster_indices[cluster_idx]) <= 1:
                continue
            random_idx= self.random_state.randint(len(cluster_indices[cluster_idx]))
            sample_a= X_min[cluster_indices[cluster_idx]][random_idx]
            dist, indices= cluster_nns[cluster_idx].kneighbors(sample_a.reshape(1, -1))
            sample_b_idx= self.random_state.choice(cluster_indices[cluster_idx][indices[0][1:]])
            sample_b= X_min[sample_b_idx]
            samples.append(self.sample_between_points(sample_a, sample_b))
            
        return np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(self.minority_label,len(samples))])
        
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion, 
                'n_neighbors': self.n_neighbors, 
                'n_clusters': self.n_clusters, 
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}



class MulticlassOversampling(StatisticsMixin):
    """
    Carries out multiclass oversampling
    
    Example::
        
        import smote_variants as sv
        import sklearn.datasets as datasets
        
        dataset= datasets.load_wine()
        
        oversampler= sv.MulticlassOversampling(sv.distance_SMOTE())
        
        X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])
    """
    
    def __init__(self, oversampler= SMOTE(random_state= 2), strategy= "equalize_1_vs_many_successive"):
        """
        Constructor of the multiclass oversampling object
        
        Args:
            oversampler (obj): an oversampling object
            strategy (str/obj): a multiclass oversampling strategy, currently 'equalize_1_vs_many_successive'/'equalize_1_vs_many'
        """
        self.oversampler= oversampler
        self.strategy= strategy
    
    def sample_equalize_1_vs_many(self, X, y):
        """
        Does the sample generation by oversampling each minority class to the
        cardinality of the majority class using all original samples in each run.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        
        # _logger.info(self.__class__.__name__ + ": " +"Running multiclass oversampling with strategy %s" % str(self.strategy))
        
        if not 'proportion' in self.oversampler.get_params():
            raise ValueError("Multiclass oversampling strategy %s cannot be used with oversampling techniques without proportion parameter" % str(self.strategy))
        
        # extract class label statistics
        self.class_label_statistics(X, y)
        
        # sort labels by number of samples
        class_labels= self.class_stats.keys()
        class_labels= sorted(class_labels, key= lambda x: -self.class_stats[x])
        
        majority_class_label= class_labels[0]
        
        # determining the majority class data
        X_maj= X[y == majority_class_label]
        
        # dict to store the results
        results= {}
        results[majority_class_label]= X_maj.copy()
        
        # running oversampling for all minority classes against all oversampled classes
        for i in range(1, len(class_labels)):
            # _logger.info(self.__class__.__name__ + ": " + ("Sampling minority class with label: %d" % class_labels[i]))
            
            # extract current minority class
            minority_class_label= class_labels[i]
            X_min= X[y == minority_class_label]
            X_maj= X[y != minority_class_label]
            
            # prepare data to pass to oversampling
            X_training= np.vstack([X_maj, X_min])
            y_training= np.hstack([np.repeat(0, len(X_maj)), np.repeat(1, len(X_min))])
            
            # prepare parameters by properly setting the proportion value
            params= self.oversampler.get_params()
            
            num_to_generate= self.class_stats[majority_class_label] - self.class_stats[class_labels[i]]
            num_to_gen_to_all= len(X_maj) - self.class_stats[class_labels[i]]
            
            params['proportion']= num_to_generate/num_to_gen_to_all
            
            # instantiating new oversampling object with the proper proportion parameter
            oversampler= self.oversampler.__class__(**params)
            
            # executing the sampling
            X_samp, y_samp= oversampler.sample(X_training, y_training)
            
            # registaring the newly oversampled minority class in the output set
            results[class_labels[i]]= X_samp[len(X_training):][y_samp[len(X_training):] == 1]
        
        # constructing the output set
        X_final= results[class_labels[1]]
        y_final= np.repeat(class_labels[1], len(results[class_labels[1]]))
        
        for i in range(2, len(class_labels)):
            X_final= np.vstack([X_final, results[class_labels[i]]])
            y_final= np.hstack([y_final, np.repeat(class_labels[i], len(results[class_labels[i]]))])
        
        return np.vstack([X, X_final]), np.hstack([y, y_final])
    
    def sample_equalize_1_vs_many_successive(self, X, y):
        """
        Does the sample generation by oversampling each minority class successively to the
        cardinality of the majority class, incorporating the results of previous
        oversamplings.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        
        # _logger.info(self.__class__.__name__ + ": " +"Running multiclass oversampling with strategy %s" % str(self.strategy))
        
        if not 'proportion' in self.oversampler.get_params():
            raise ValueError("Multiclass oversampling strategy %s cannot be used with oversampling techniques without proportion parameter" % str(self.strategy))
        
        # extract class label statistics
        self.class_label_statistics(X, y)
        
        # sort labels by number of samples
        class_labels= self.class_stats.keys()
        class_labels= sorted(class_labels, key= lambda x: -self.class_stats[x])
        
        majority_class_label= class_labels[0]
        
        # determining the majority class data
        X_maj= X[y == majority_class_label]
        
        # dict to store the results
        results= {}
        results[majority_class_label]= X_maj.copy()
        
        # running oversampling for all minority classes against all oversampled classes
        for i in range(1, len(class_labels)):
            # _logger.info(self.__class__.__name__ + ": " + ("Sampling minority class with label: %d" % class_labels[i]))
            
            # extract current minority class
            minority_class_label= class_labels[i]
            X_min= X[y == minority_class_label]
            
            # prepare data to pass to oversampling
            X_training= np.vstack([X_maj, X_min])
            y_training= np.hstack([np.repeat(0, len(X_maj)), np.repeat(1, len(X_min))])
            
            # prepare parameters by properly setting the proportion value
            params= self.oversampler.get_params()
            
            num_to_generate= self.class_stats[majority_class_label] - self.class_stats[class_labels[i]]
            num_to_gen_to_all= (i*self.class_stats[majority_class_label] - self.class_stats[class_labels[i]])
            
            params['proportion']= num_to_generate/num_to_gen_to_all
            
            # instantiating new oversampling object with the proper proportion parameter
            oversampler= self.oversampler.__class__(**params)
            
            # executing the sampling
            X_samp, y_samp= oversampler.sample(X_training, y_training)
            
            # adding the newly oversampled minority class to the majority data
            X_maj= np.vstack([X_maj, X_samp[y_samp == 1]])
            
            # registaring the newly oversampled minority class in the output set
            results[class_labels[i]]= X_samp[len(X_training):][y_samp[len(X_training):] == 1]

        # constructing the output set        
        X_final= results[class_labels[1]]
        y_final= np.repeat(class_labels[1], len(results[class_labels[1]]))
        
        for i in range(2, len(class_labels)):
            X_final= np.vstack([X_final, results[class_labels[i]]])
            y_final= np.hstack([y_final, np.repeat(class_labels[i], len(results[class_labels[i]]))])
        
        return np.vstack([X, X_final]), np.hstack([y, y_final])
        
    def sample(self, X, y):
        """
        Does the sample generation according to the oversampling strategy.
        
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
            
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        
        if self.strategy == "equalize_1_vs_many_successive":
            return self.sample_equalize_1_vs_many_successive(X, y)
        elif self.strategy == "equalize_1_vs_many":
            return self.sample_equalize_1_vs_many(X, y)
        else:
            raise ValueError("Multiclass oversampling startegy %s not implemented." % self.strategy)
    
    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the multiclass oversampling object
        """
        return {'oversampler': self.oversampler, 'strategy': self.strategy}

class OversamplingClassifier(BaseEstimator, ClassifierMixin):
    """
    This class wraps an oversampler and a classifier, making it compatible
    with sklearn based pipelines.
    """
    
    def __init__(self, oversampler, classifier):
        """
        Constructor of the wrapper.
        
        Args:
            oversampler (obj): an oversampler object
            classifier (obj): an sklearn-compatible classifier
        """
        
        self.oversampler= oversampler
        self.classifier= classifier
        
    def fit(self, X, y=None):
        """
        Carries out oversampling and fits the classifier.
        
        Args:
            X (np.ndarray): feature vectors
            y (np.array): target values
        
        Returns:
            obj: the object itself
        """
        
        X_samp, y_samp= self.oversampler.sample(X, y)
        self.classifier.fit(X_samp, y_samp)
        
        return self
    
    def predict(self, X):
        """
        Carries out the predictions.
        
        Args:
            X (np.ndarray): feature vectors
        """
        
        return self.classifier.predict(X)
    
    def predict_proba(self, X):
        """
        Carries out the predictions with probability estimations.
        
        Args:
            X (np.ndarray): feature vectors
        """
        
        return self.classifier.predict_proba(X)
    
    def get_params(self, deep=True):
        """
        Returns the dictionary of parameters.
        
        Args:
            deep (bool): wether to return parameters with deep discovery
        
        Returns:
            dict: the dictionary of parameters
        """
        
        return {'oversampler': self.oversampler, 'classifier': self.classifier}
    
    def set_params(self, **parameters):
        """
        Sets the parameters.
        
        Args:
            parameters (dict): the parameters to set.
        
        Returns:
            obj: the object itself
        """
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        return self

class MLPClassifierWrapper:
    """
    Wrapper over MLPClassifier of sklearn to provide easier parameterization
    """
    def __init__(self, activation= 'relu', hidden_layer_fraction= 0.1, alpha= 0.0001, random_state= None):
        """
        Constructor of the MLPClassifier
        
        Args:
            activation (str): name of the activation function
            hidden_layer_fraction (float): fraction of the hidden neurons of the number of input dimensions
            alpha (float): alpha parameter of the MLP classifier
            random_state (int/np.random.RandomState/None): initializer of the random state
        """
        self.activation= activation
        self.hidden_layer_fraction= hidden_layer_fraction
        self.alpha= alpha
        self.random_state= random_state
    
    def fit(self, X, y):
        """
        Fit the model to the data
        
        Args:
            X (np.ndarray): features
            y (np.array): target labels
            
        Returns:
            obj: the MLPClassifierWrapper object
        """
        hidden_layer_size= max([1, int(len(X[0])*self.hidden_layer_fraction)])
        self.model= MLPClassifier(activation= self.activation, 
                                  hidden_layer_sizes= (hidden_layer_size,),
                                  alpha= self.alpha,
                                  random_state= self.random_state).fit(X, y)
        return self
        
    def predict(self, X):
        """
        Predicts the labels of the unseen data
        
        Args:
            X (np.ndarray): unseen features
            
        Returns:
            np.array: predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predicts the class probabilities of the unseen data
        
        Args:
            X (np.ndarray): unseen features
            
        Returns:
            np.matrix: predicted class probabilities
        """
        return self.model.predict_proba(X)
    
    def get_params(self, deep=False):
        """
        Returns the parameters of the classifier.
        
        Returns:
            dict: the parameters of the object
        """
        return {'activation': self.activation, 'hidden_layer_fraction': self.hidden_layer_fraction, 'alpha': self.alpha, 'random_state': self.random_state}
    
    def copy(self):
        """
        Creates a copy of the classifier.
        
        Returns:
            obj: a copy of the classifier
        """
        return MLPClassifierWrapper(**self.get_params())

class Folding():
    """
    Cache-able folding of dataset for cross-validation
    """
    def __init__(self, dataset, validator, cache_path= None, random_state= None):
        """
        Constructor of Folding object
        
        Args:
            dataset (dict): dataset dictionary with keys 'data', 'target' and 'DESCR'
            validator (obj): cross-validator object
            cache_path (str): path to cache directory
            random_state (int/np.random.RandomState/None): initializer of the random state
        """
        self.dataset= dataset
        self.db_name= self.dataset['name']
        self.validator= validator
        self.cache_path= cache_path
        self.filename= 'folding_' + self.db_name + '.pickle'
        self.db_size= len(dataset['data'])
        self.db_n_attr= len(dataset['data'][0])
        self.imbalanced_ratio= np.sum(self.dataset['target'] == 0)/np.sum(self.dataset['target'] == 1)
        self.random_state= random_state
    
    def do_folding(self):
        """
        Does the folding or reads it from file if already available
        
        Returns:
            list(tuple): list of tuples of X_train, y_train, X_test, y_test objects
        """
        
        self.validator.random_state= self.random_state
        
        if not hasattr(self, 'folding'):
            if (self.cache_path is None) or (not self.cache_path is None) and (not os.path.isfile(os.path.join(self.cache_path, self.filename))):
                # _logger.info(self.__class__.__name__ + (" doing folding %s" % self.filename))
                
                self.folding= {}
                self.folding['folding']= []
                self.folding['db_size']= len(self.dataset['data'])
                self.folding['db_n_attr']= len(self.dataset['data'][0])
                self.folding['imbalanced_ratio']= np.sum(self.dataset['target'] == 0)/np.sum(self.dataset['target'] == 1)
                
                X= self.dataset['data']
                y= self.dataset['target']
                for train, test in self.validator.split(self.dataset['data'], self.dataset['target'], self.dataset['target']):
                    self.folding['folding'].append((X[train], y[train], X[test], y[test]))
                if not self.cache_path is None:
                    # _logger.info(self.__class__.__name__ + (" dumping to file %s" % self.filename))
                    pickle.dump(self.folding, open(os.path.join(self.cache_path, self.filename), "wb"))
            else:
                # _logger.info(self.__class__.__name__ + (" reading from file %s" % self.filename))
                self.folding= pickle.load(open(os.path.join(self.cache_path, self.filename), "rb"))
        return self.folding
    
    def get_params(self, deep=False):
        return {'db_name': self.db_name}
    
    def descriptor(self):
        return str(self.get_params())
        
class Sampling():
    """
    Cache-able sampling of dataset folds
    """
    def __init__(self, folding, sampler, sampler_parameters, scaler, random_state= None):
        """
        Constructor of the sampling object
        
        Args:
            folding (obj): Folding object
            sampler (class): class of a sampler object
            sampler_parameters (dict): a parameter combination for the sampler object
            scaler (obj): scaler object
            random_state (int/np.random.RandomState/None): initializer of the random state
        """
        self.folding= folding
        self.db_name= folding.db_name
        self.sampler= sampler
        self.sampler_parameters= sampler_parameters
        self.scaler= scaler
        self.cache_path= folding.cache_path
        self.filename= self.standardized_filename('sampling')
        self.random_state= random_state
        
    def standardized_filename(self, prefix, db_name= None, sampler= None, sampler_parameters= None):
        """
        standardizes the filename
        
        Args:
            filename (str): filename
            
        Returns:
            str: standardized name
        """
        import hashlib
        
        db_name= (db_name or self.db_name)
        
        sampler= (sampler or self.sampler.__name__)
        sampler_parameters= sampler_parameters or self.sampler_parameters
        sampler_parameter_str= hashlib.md5(str(sampler_parameters).encode('utf-8')).hexdigest()
        
        filename= '_'.join([prefix, db_name, sampler, sampler_parameter_str]) + '.pickle'
        filename= re.sub('["\\,:(){}]', '', filename)
        filename= filename.replace("'", '')
        filename= filename.replace(": ", "_")
        filename= filename.replace(" ", "_")
        filename= filename.replace("\n", "_")
        
        return filename
    
    
    def cache_sampling(self):
        try:
            import mkl
            mkl.set_num_threads(1)
            # _logger.info(self.__class__.__name__ + (" mkl thread number set to 1 successfully"))
        except:pass
            # _logger.info(self.__class__.__name__ + (" setting mkl thread number didn't succeed"))
        
        if not os.path.isfile(os.path.join(self.cache_path, self.filename)):
            # if the sampled dataset does not exist
            is_extensive= OverSampling.cat_extensive in self.sampler.categories
            has_proportion= 'proportion' in self.sampler_parameters
            higher_prop_sampling_available= None
            
            if is_extensive and has_proportion:
                proportion= self.sampler_parameters['proportion']
                all_pc= self.sampler.parameter_combinations()
                all_proportions= np.unique([p['proportion'] for p in all_pc])
                all_proportions= all_proportions[all_proportions > proportion]
                
                for p in all_proportions:
                    tmp_par= self.sampler_parameters.copy()
                    tmp_par['proportion']= p
                    tmp_filename= self.standardized_filename('sampling', self.db_name, str(self.sampler.__name__), str(tmp_par))
                    
                    if os.path.isfile(os.path.join(self.cache_path, tmp_filename)):
                        higher_prop_sampling_available= (p, tmp_filename)
                        break
            
            if not is_extensive or not has_proportion or (is_extensive and has_proportion and higher_prop_sampling_available is None):
                # _logger.info(self.__class__.__name__ + " doing sampling")
                begin= time.time()
                sampling= []
                folds= self.folding.do_folding()
                self.sampler_parameters['random_state']= self.random_state
                for X_train, y_train, X_test, y_test in folds['folding']:
                    s= self.sampler(**self.sampler_parameters)
                    # print('scaler', self.scaler)
                    #if not self.scaler is None and not hasattr(s, 'transform'):
                    if not self.scaler is None:
                        X_train= self.scaler.fit_transform(X_train, y_train)
                    X_samp, y_samp= s.sample_with_timing(X_train, y_train)
                    
                    if hasattr(s, 'transform'):
                        X_test_trans= s.transform(X_test)
                    else:
                        X_test_trans= X_test.copy()
                    
                    #if not self.scaler is None and not hasattr(s, 'transform'):
                    if not self.scaler is None:
                        X_samp= self.scaler.inverse_transform(X_samp)
                    
                    sampling.append((X_samp, y_samp, X_test_trans, y_test))
                runtime= time.time() - begin
            else:
                higher_prop, higher_prop_filename= higher_prop_sampling_available
                # _logger.info(self.__class__.__name__ + (" reading and resampling from file %s to %s" % (higher_prop_filename, self.filename)))
                tmp_results= pickle.load(open(os.path.join(self.cache_path, higher_prop_filename), 'rb'))
                tmp_sampling= tmp_results['sampling']
                tmp_runtime= tmp_results['runtime']
                
                sampling= []
                folds= self.folding.do_folding()
                nums= [len(X_train) for X_train, _, _, _ in folds['folding']]
                i= 0
                for X_train, y_train, X_test, y_test in tmp_sampling:
                    new_num= int((len(X_train) - nums[i])/higher_prop*proportion)
                    sampling.append((X_train[:(nums[i] + new_num)], y_train[:(nums[i] + new_num)], X_test, y_test))
                    i= i + 1
                runtime= tmp_runtime/p*proportion
                
            results= {}
            results['sampling']= sampling
            results['runtime']= runtime
            results['db_size']= folds['db_size']
            results['db_n_attr']= folds['db_n_attr']
            results['imbalanced_ratio']= folds['imbalanced_ratio']
            
            # _logger.info(self.__class__.__name__ + (" dumping to file %s" % self.filename))
            pickle.dump(results, open(os.path.join(self.cache_path, self.filename), "wb"))
        
    def do_sampling(self):
        self.cache_sampling()
        results= pickle.load(open(os.path.join(self.cache_path, self.filename), 'rb'))
        return results
    
    def get_params(self, deep=False):
        return {'folding': self.folding.get_params(), 'sampler_name': self.sampler.__name__, 'sampler_parameters': self.sampler_parameters}
    
    def descriptor(self):
        return str(self.get_params())

class Evaluation():
    """
    Cache-able evaluation of classifier on sampling
    """
    def __init__(self, sampling, classifiers, n_threads= None, random_state= None):
        """
        Constructor of an Evaluation object
        
        Args:
            sampling (obj): Sampling object
            classifiers (list(obj)): classifier objects
            n_threads (int/None): number of threads
            random_state (int/np.random.RandomState/None): random state initializer
        """
        self.sampling= sampling
        self.classifiers= classifiers
        self.n_threads= n_threads
        self.cache_path= sampling.cache_path
        self.filename= self.sampling.standardized_filename('eval')
        self.random_state= random_state
        
        self.labels= []
        for i in range(len(classifiers)):
            label= str((self.sampling.get_params(), classifiers[i].__class__.__name__, classifiers[i].get_params()))
            self.labels.append(label)
    
    def calculate_metrics(self, all_pred, all_test):
        """
        Calculates metrics of binary classifiction
        
        Args:
            all_pred (np.matrix): predicted probabilities
            all_test (np.matrix): true labels
            
        Returns:
            dict: all metrics of binary classification
        """

        results= {}
        if not all_pred is None:
            all_pred_labels= np.apply_along_axis(lambda x: np.argmax(x), 1, all_pred)
    
            results['tp']= np.sum(np.logical_and(np.equal(all_test, all_pred_labels), (all_test == 1)))
            results['tn']= np.sum(np.logical_and(np.equal(all_test, all_pred_labels), (all_test == 0)))
            results['fp']= np.sum(np.logical_and(np.logical_not(np.equal(all_test, all_pred_labels)), (all_test == 0)))
            results['fn']= np.sum(np.logical_and(np.logical_not(np.equal(all_test, all_pred_labels)), (all_test == 1)))
            results['p']= results['tp'] + results['fn']
            results['n']= results['fp'] + results['tn']
            results['acc']= (results['tp'] + results['tn'])/(results['p'] + results['n'])
            results['sens']= results['tp']/results['p']
            results['spec']= results['tn']/results['n']
            results['ppv']= results['tp']/(results['tp'] + results['fp'])
            results['npv']= results['tn']/(results['tn'] + results['fn'])
            results['fpr']= 1.0 - results['spec']
            results['fdr']= 1.0 - results['ppv']
            results['fnr']= 1.0 - results['sens']
            results['bacc']= (results['tp']/results['p'] + results['tn']/results['n'])/2.0
            results['gacc']= np.sqrt(results['tp']/results['p']*results['tn']/results['n'])
            results['f1']= 2*results['tp']/(2*results['tp'] + results['fp'] + results['fn'])
            results['mcc']= (results['tp']*results['tn'] - results['fp']*results['fn'])/np.sqrt((results['tp'] + results['fp'])*(results['tp'] + results['fn'])*(results['tn'] + results['fp'])*(results['tn'] + results['fn']))
            results['l']= (results['p'] + results['n'])*np.log(results['p'] + results['n'])
            results['ltp']= results['tp']*np.log(results['tp']/((results['tp'] + results['fp'])*(results['tp'] + results['fn'])))
            results['lfp']= results['fp']*np.log(results['fp']/((results['fp'] + results['tp'])*(results['fp'] + results['tn'])))
            results['lfn']= results['fn']*np.log(results['fn']/((results['fn'] + results['tp'])*(results['fn'] + results['tn'])))
            results['ltn']= results['tn']*np.log(results['tn']/((results['tn'] + results['fp'])*(results['tn'] + results['fn'])))
            results['lp']= results['p']*np.log(results['p']/(results['p'] + results['n']))
            results['ln']= results['n']*np.log(results['n']/(results['p'] + results['n']))
            results['uc']= (results['l'] + results['ltp'] + results['lfp'] + results['lfn'] + results['ltn'])/(results['l'] + results['lp'] + results['ln'])
            results['informedness']= results['sens'] + results['spec'] - 1.0
            results['markedness']= results['ppv'] + results['npv'] - 1.0
            results['log_loss']= log_loss(all_test, all_pred)
            results['auc']= roc_auc_score(all_test, all_pred[:,1])
            test_labels, preds= zip(*sorted(zip(all_test, all_pred[:,1]), key= lambda x: -x[1]))
            test_labels= np.array(test_labels)
            th= int(0.2*len(test_labels))
            results['p_top20']= np.sum(test_labels[:th] == 1)/th
            results['brier']= np.mean((all_pred[:,1] - all_test)**2)
        else:
            results['tp']= 0
            results['tn']= 0
            results['fp']= 0
            results['fn']= 0
            results['p']= 0
            results['n']= 0
            results['acc']= 0
            results['sens']= 0
            results['spec']= 0
            results['ppv']= 0
            results['npv']= 0
            results['fpr']= 1
            results['fdr']= 1
            results['fnr']= 1
            results['bacc']= 0
            results['gacc']= 0
            results['f1']= 0
            results['mcc']= np.nan
            results['l']= np.nan
            results['ltp']= np.nan
            results['lfp']= np.nan
            results['lfn']= np.nan
            results['ltn']= np.nan
            results['lp']= np.nan
            results['ln']= np.nan
            results['uc']= np.nan
            results['informedness']= 0
            results['markedness']= 0
            results['log_loss']= np.nan
            results['auc']= 0
            results['p_top20']= 0
            results['brier']= 1
        
        return results
    
    def do_evaluation(self):
        """
        Does the evaluation or reads it from file
        
        Returns:
            dict: all metrics
        """
        
        if not self.n_threads is None:
            try:
                import mkl
                mkl.set_num_threads(self.n_threads)
            except:pass
        
        evaluations= {}
        if os.path.isfile(os.path.join(self.cache_path, self.filename)):
            evaluations= pickle.load(open(os.path.join(self.cache_path, self.filename), 'rb'))
        
        already_evaluated= np.array([l in evaluations for l in self.labels])
        
        if not np.all(already_evaluated):
            samp= self.sampling.do_sampling()
        else:
            return list(evaluations.values())
        
        # setting random states
        for i in range(len(self.classifiers)):
            clf_params= self.classifiers[i].get_params()
            if 'random_state' in clf_params:
                clf_params['random_state']= self.random_state
                self.classifiers[i]= self.classifiers[i].__class__(**clf_params)
            if isinstance(self.classifiers[i], CalibratedClassifierCV):
                clf_params= self.classifiers[i].base_estimator.get_params()
                clf_params['random_state']= self.random_state
                self.classifiers[i].base_estimator= self.classifiers[i].base_estimator.__class__(**clf_params)
        
        for i in range(len(self.classifiers)):
            if not already_evaluated[i]:
                all_preds, all_tests= [], []
                minority_class_label= None
                majority_class_label= None
                for X_train, y_train, X_test, y_test in samp['sampling']:
                    class_labels= np.unique(y_train)
                    min_class_size= np.min([np.sum(y_train == c) for c in class_labels])
                    
                    ss= StandardScaler()
                    X_train_trans= ss.fit_transform(X_train)
                    nonzero_var_idx= np.where(ss.var_ > 1e-8)[0]
                    X_test_trans= ss.transform(X_test)
                    
                    enough_minority_samples= min_class_size > 4
                    y_train_big_enough= len(y_train) > 4
                    two_classes= len(class_labels) > 1
                    at_least_one_feature= (len(nonzero_var_idx) > 0)
                    
                    if not enough_minority_samples:pass
                    elif not y_train_big_enough:pass
                    elif not two_classes:pass
                    elif not at_least_one_feature:pass
                    else:
                        all_tests.append(y_test)
                        if minority_class_label is None or majority_class_label is None:
                            class_labels= np.unique(y_train)
                            if sum(class_labels[0] == y_test) < sum(class_labels[1] == y_test):
                                minority_class_label= int(class_labels[0])
                                majority_class_label= int(class_labels[1])
                            else:
                                minority_class_label= int(class_labels[1])
                                majority_class_label= int(class_labels[0])
                        
                        self.classifiers[i].fit(X_train_trans[:,nonzero_var_idx], y_train)
                        all_preds.append(self.classifiers[i].predict_proba(X_test_trans[:,nonzero_var_idx]))
                
                if len(all_tests) > 0:
                    all_preds= np.vstack(all_preds)
                    all_tests= np.hstack(all_tests)
                    
                    evaluations[self.labels[i]]= self.calculate_metrics(all_preds, all_tests)
                else:
                    evaluations[self.labels[i]]= self.calculate_metrics(None, None)
                    
                evaluations[self.labels[i]]['runtime']= samp['runtime']
                evaluations[self.labels[i]]['sampler']= self.sampling.sampler.__name__
                evaluations[self.labels[i]]['classifier']= self.classifiers[i].__class__.__name__
                evaluations[self.labels[i]]['sampler_parameters']= str(self.sampling.sampler_parameters)
                evaluations[self.labels[i]]['classifier_parameters']= str(self.classifiers[i].get_params())
                evaluations[self.labels[i]]['sampler_categories']= str(self.sampling.sampler.categories)
                evaluations[self.labels[i]]['db_name']= self.sampling.folding.db_name
                evaluations[self.labels[i]]['db_size']= samp['db_size']
                evaluations[self.labels[i]]['db_n_attr']= samp['db_n_attr']
                evaluations[self.labels[i]]['imbalanced_ratio']= samp['imbalanced_ratio']

        if not np.all(already_evaluated):
            # _logger.info(self.__class__.__name__ + (" dumping to file %s" % self.filename))
            pickle.dump(evaluations, open(os.path.join(self.cache_path, self.filename), "wb"))

        return list(evaluations.values())

def trans(X):
    """
    Transformation function used to aggregate the evaluation results.
    
    Args:
        X (pd.DataFrame): a grouping of a data frame containing evaluation results
    """
    return pd.DataFrame({'auc': np.max(X['auc']), 
                         'brier': np.min(X['brier']), 
                         'acc': np.max(X['acc']), 
                         'f1': np.max(X['f1']),
                         'p_top20': np.max(X['p_top20']), 
                         'gacc': np.max(X['gacc']), 
                         'runtime': np.mean(X['runtime']),
                         'db_size': X['db_size'].iloc[0], 
                         'db_n_attr': X['db_n_attr'].iloc[0], 
                         'imbalanced_ratio': X['imbalanced_ratio'].iloc[0],
                         'sampler_categories': X['sampler_categories'].iloc[0], 
                         'classifier_parameters_auc': X.sort_values('auc')['classifier_parameters'].iloc[-1],
                         'classifier_parameters_acc': X.sort_values('acc')['classifier_parameters'].iloc[-1],
                         'classifier_parameters_gacc': X.sort_values('gacc')['classifier_parameters'].iloc[-1],
                         'classifier_parameters_f1': X.sort_values('f1')['classifier_parameters'].iloc[-1],
                         'classifier_parameters_p_top20': X.sort_values('p_top20')['classifier_parameters'].iloc[-1],
                         'classifier_parameters_brier': X.sort_values('brier')['classifier_parameters'].iloc[-1],
                         'sampler_parameters_auc': X.sort_values('auc')['sampler_parameters'].iloc[-1],
                         'sampler_parameters_acc': X.sort_values('acc')['sampler_parameters'].iloc[-1],
                         'sampler_parameters_gacc': X.sort_values('gacc')['sampler_parameters'].iloc[-1],
                         'sampler_parameters_f1': X.sort_values('f1')['sampler_parameters'].iloc[-1],
                         'sampler_parameters_p_top20': X.sort_values('p_top20')['sampler_parameters'].iloc[-1],
                         'sampler_parameters_brier': X.sort_values('p_top20')['sampler_parameters'].iloc[0],
                         }, index= [0])

def _clone_classifiers(classifiers):
    """
    Clones a set of classifiers
    
    Args:
        classifiers (list): a list of classifier objects
    """
    results= []
    for c in classifiers:
        if isinstance(c, MLPClassifierWrapper):
            results.append(c.copy())
        else:
            results.append(clone(c))

    return results
    
def _cache_samplings(folding, samplers, scaler, max_n_sampler_par_comb= 35, n_jobs= 1, random_state= None): 
    """
    
    """
    # _logger.info("create sampling objects")
    sampling_objs= []
    
    if isinstance(random_state, int):
        random_state= np.random.RandomState(random_state)
    elif random_state is None:
        random_state= np.random
    
    for s in samplers:
    
        sampling_par_comb= s.parameter_combinations()
        sampling_par_comb= random_state.choice(sampling_par_comb, min([len(sampling_par_comb), max_n_sampler_par_comb]), replace= False)
        
        for spc in sampling_par_comb:
            sampling_objs.append(Sampling(folding, s, spc, scaler, random_state))
            
    # sorting sampling objects to optimize execution
    def key(x):
        if isinstance(x.sampler, ADG) or isinstance(x.sampler, AMSCO) or isinstance(x.sampler, DSRBF):
            if 'proportion' in x.sampler_parameters:
                return 30 + x.sampler_parameters['proportion']
            else:
                return 30
        elif 'proportion' in x.sampler_parameters:
            return x.sampler_parameters['proportion']
        elif OverSampling.cat_memetic in x.sampler.categories:
            return 20
        else:
            return 10
    
    sampling_objs= list(reversed(sorted(sampling_objs, key= key)))
    
    # executing sampling in parallel
    # _logger.info("executing %d sampling in parallel" % len(sampling_objs))
    Parallel(n_jobs= n_jobs, batch_size= 1)(delayed(s.cache_sampling)() for s in sampling_objs)
    
    return sampling_objs
            
def _cache_evaluations(sampling_objs, classifiers, n_jobs= 1, random_state= None):
    # create evaluation objects
    # _logger.info("create classifier jobs")
    evaluation_objs= []
    
    num_threads = None if n_jobs is None or n_jobs is 1 else 1
    
    for s in sampling_objs:
        evaluation_objs.append(Evaluation(s, _clone_classifiers(classifiers), num_threads, random_state))
    
    # _logger.info("executing %d evaluation jobs in parallel" % (len(evaluation_objs)))
    # execute evaluation in parallel
    evals= Parallel(n_jobs= n_jobs, batch_size= 1)(delayed(e.do_evaluation)() for e in evaluation_objs)
    
    return evals

def _read_db_results(cache_path_db):
    results= []
    evaluation_files= glob.glob(os.path.join(cache_path_db, 'eval*.pickle'))
    
    for f in evaluation_files:
        eval_results= pickle.load(open(f, 'rb'))
        results.append(list(eval_results.values()))
    
    return results

def read_oversampling_results(datasets, cache_path= None, all_results= False):
    """
    Reads the results of the evaluation
    
    Args:
        datasets (list): list of datasets and/or dataset loaders - a dataset is a dict with 'data', 'target' and 'name' keys
        cache_path (str): path to a cache directory
        all_results (bool): True to return all results, False to return an aggregation
        
    Returns:
        pd.DataFrame: all results or the aggregated results if all_results is False
    """
    
    results= []
    for dataset_spec in datasets:
        
        # loading dataset if needed and determining dataset name
        dataset= dataset_spec() if not isinstance(dataset_spec, dict) else dataset_spec
        dataset_name= dataset['name'] if 'name' in dataset else dataset_spec.__name__
        dataset['name']= dataset_name

        # determining dataset specific cache path
        cache_path_db= os.path.join(cache_path, dataset_name)
        
        # reading the results
        res= _read_db_results(cache_path_db)
        
        # concatenating the results
        # _logger.info("concatenating results")
        db_res= [pd.DataFrame(r) for r in res]
        db_res= pd.concat(db_res).reset_index(drop= True)
        
        # _logger.info("aggregating the results")
        if all_results == False:
            db_res= db_res.groupby(by= ['db_name', 'classifier', 'sampler']).apply(trans).reset_index().drop('level_3', axis= 1)

        results.append(db_res)
    
    return pd.concat(results).reset_index(drop= True)
    
def evaluate_oversamplers(datasets,
                          samplers,
                          classifiers,
                          cache_path,
                          validator= RepeatedStratifiedKFold(n_splits= 5, n_repeats= 3),
                          scaler= None,
                          all_results= False, 
                          remove_sampling_cache= False, 
                          max_samp_par_comb= 35,
                          n_jobs= 1,
                          random_state= None):
    """
    Evaluates oversampling techniques using various classifiers on various datasets
    
    Args:
        datasets (list): list of datasets and/or dataset loaders - a dataset is a dict with 'data', 'target' and 'name' keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        cache_path (str): path to a cache directory
        validator (obj): validator object
        scaler (obj): scaler object
        all_results (bool): True to return all results, False to return an aggregation
        remove_sampling_cache (bool): True to remove sampling objects after evaluation
        max_samp_par_comb (int): maximum number of sampler parameter combinations to be tested
        n_jobs (int): number of parallel jobs
        random_state (int/np.random.RandomState/None): initializer of the random state
        
    Returns:
        pd.DataFrame: all results or the aggregated results if all_results is False
        
    Example::
        
        import smote_variants as sv
        import imbalanced_datasets as imbd
        
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        
        datasets= [imbd.load_glass2, imbd.load_ecoli4]
        oversamplers= [sv.SMOTE_ENN, sv.NEATER, sv.Lee]
        classifiers= [KNeighborsClassifier(n_neighbors= 3),
                      KNeighborsClassifier(n_neighbors= 5),
                      DecisionTreeClassifier()]
        
        cache_path= '/home/<user>/smote_validation/'
        
        results= evaluate_oversamplers(datasets,
                                       oversamplers,
                                       classifiers,
                                       cache_path)
    """
    
    if cache_path is None:
        raise ValueError('cache_path is not specified')
    
    results= []
    for dataset_spec in datasets:
        # loading dataset if needed and determining dataset name
        dataset= dataset_spec() if not isinstance(dataset_spec, dict) else dataset_spec
        dataset_name= dataset['name'] if 'name' in dataset else dataset_spec.__name__
        dataset['name']= dataset_name
        
        dataset_original_target= dataset['target'].copy()
        class_labels= np.unique(dataset['target'])
        if sum(dataset['target'] == class_labels[0]) < sum(dataset['target'] == class_labels[1]):
            min_label= class_labels[0]
            maj_label= class_labels[1]
        else:
            min_label= class_labels[1]
            maj_label= class_labels[0]
        min_ind= np.where(dataset['target'] == min_label)[0]
        maj_ind= np.where(dataset['target'] == maj_label)[0]
        np.put(dataset['target'], min_ind, 1)
        np.put(dataset['target'], maj_ind, 0)

        cache_path_db= os.path.join(cache_path, dataset_name)
        if not os.path.isdir(cache_path_db):
            # _logger.info("creating cache directory")
            os.makedirs(cache_path_db)
        
        # checking of samplings and evaluations are available
        samplings_available= False
        evaluations_available= False
        
        samplings= glob.glob(os.path.join(cache_path_db, 'sampling*.pickle'))
        if len(samplings) > 0:
            samplings_available= True
            
        evaluations= glob.glob(os.path.join(cache_path_db, 'eval*.pickle'))
        if len(evaluations) > 0:
            evaluations_available= True
        
        # _logger.info("dataset: %s, samplings_available: %s, evaluations_available: %s" % (dataset_name, str(samplings_available), str(evaluations_available)))

        if remove_sampling_cache and evaluations_available and not samplings_available:
            # remove_sampling_cache is enabled and evaluations are available, they are being read
            # _logger.info("reading result from cache, sampling and evaluation is not executed")
            res= _read_db_results(cache_path_db)
        else:
            # _logger.info("doing the folding")
            folding= Folding(dataset, validator, cache_path_db, random_state)
            folding.do_folding()
            
            # _logger.info("do the samplings")
            sampling_objs= _cache_samplings(folding, samplers, scaler, max_samp_par_comb, n_jobs, random_state)
            
            # _logger.info("do the evaluations")
            res= _cache_evaluations(sampling_objs, classifiers, n_jobs, random_state)
        
        dataset['target']= dataset_original_target
        
        # removing samplings once everything is done
        if remove_sampling_cache:
            filenames= glob.glob(os.path.join(cache_path_db, 'sampling*'))
            # _logger.info("removing unnecessary sampling files")
            if len(filenames) > 0:
                for f in filenames:
                    os.remove(f)
        
        # _logger.info("concatenating the results")
        db_res= [pd.DataFrame(r) for r in res]
        db_res= pd.concat(db_res).reset_index(drop= True)
        
        #def filter_results(x):
        #    if "'p'" in x and 'p' in eval(x) and eval(x)['p'] == 2:
        #        return True
        #    else:
        #        return False
        
        #db_res= db_res[db_res['classifier_parameters'].apply(lambda x: filter_results(x))]
        
        pickle.dump(db_res, open(os.path.join(cache_path_db, 'results.pickle'), 'wb'))
        
        # _logger.info("aggregating the results")
        if all_results == False:
            db_res= db_res.groupby(by= ['db_name', 'classifier', 'sampler']).apply(trans).reset_index().drop('level_3', axis= 1)
        
        results.append(db_res)
    
    return pd.concat(results).reset_index(drop= True)


def model_selection(dataset,
                      samplers,
                      classifiers,
                      cache_path,
                      score= 'auc',
                      validator= RepeatedStratifiedKFold(n_splits= 5, n_repeats= 3),
                      remove_sampling_cache= False, 
                      max_samp_par_comb= 35,
                      n_jobs= 1,
                      random_state= None):
    """
    Evaluates oversampling techniques on various classifiers and a dataset
    and returns the oversampling and classifier objects giving the best performance
    
    Args:
        dataset (dict): a dataset is a dict with 'data', 'target' and 'name' keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        cache_path (str): path to a cache directory
        score (str): 'auc'/'acc'/'gacc'/'f1'/'brier'/'p_top20'
        validator (obj): validator object
        all_results (bool): True to return all results, False to return an aggregation
        remove_sampling_cache (bool): True to remove sampling objects after evaluation
        max_samp_par_comb (int): maximum number of sampler parameter combinations to be tested
        n_jobs (int): number of parallel jobs
        random_state (int/np.random.RandomState/None): initializer of the random state
        
    Returns:
        obj, obj: the best performing sampler object and the best performing classifier object
        
    Example::
        
        import smote_variants as sv
        import imbalanced_datasets as imbd
        
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        
        datasets= imbd.load_glass2()
        oversamplers= [sv.SMOTE_ENN, sv.NEATER, sv.Lee]
        classifiers= [KNeighborsClassifier(n_neighbors= 3),
                      KNeighborsClassifier(n_neighbors= 5),
                      DecisionTreeClassifier()]
        
        cache_path= '/home/<user>/smote_validation/'
        
        sampler, classifier= model_selection(dataset,
                                             oversamplers,
                                             classifiers,
                                             cache_path,
                                             'auc')
    """
    
    if score not in ['auc', 'acc', 'gacc', 'f1', 'brier', 'p_top20']:
        raise ValueError("score %s not supported" % score)
    
    results= evaluate_oversamplers(datasets= [dataset],
                                   samplers= samplers,
                                   classifiers= classifiers,
                                   cache_path= cache_path,
                                   validator= validator,
                                   remove_sampling_cache= remove_sampling_cache,
                                   max_samp_par_comb= max_samp_par_comb,
                                   n_jobs= n_jobs,
                                   random_state= random_state)
    
    # extracting the best performing classifier and oversampler parameters regarding AUC
    highest_score= results[score].idxmax()
    cl_par_name= 'classifier_parameters_' + score
    samp_par_name= 'sampler_parameters_' + score
    cl, cl_par, samp, samp_par= results.loc[highest_score][['classifier',
                                                           cl_par_name,
                                                           'sampler',
                                                           samp_par_name]]
    
    # instantiating the best performing oversampler and classifier objects
    samp_obj= eval(samp)(**eval(samp_par))
    cl_obj= eval(cl)(**eval(cl_par))
    
    return samp_obj, cl_obj

def cross_validate(dataset,
                   sampler,
                   classifier,
                   validator= RepeatedStratifiedKFold(n_splits= 5, n_repeats= 3),
                   scaler= StandardScaler(),
                   random_state= None):
    """
    Evaluates oversampling techniques on various classifiers and a dataset
    and returns the oversampling and classifier objects giving the best performance
    
    Args:
        dataset (dict): a dataset is a dict with 'data', 'target' and 'name' keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        validator (obj): validator object
        scaler (obj): scaler object
        random_state (int/np.random.RandomState/None): initializer of the random state
        
    Returns:
        pd.DataFrame: the cross-validation scores
        
    Example::
        
        import smote_variants as sv
        import imbalanced_datasets as imbd
        
        from sklearn.neighbors import KNeighborsClassifier
        
        dataset= imbd.load_glass2()
        sampler= sv.SMOTE_ENN
        classifier= KNeighborsClassifier(n_neighbors= 3)
        
        sampler, classifier= model_selection(dataset,
                                             oversampler,
                                             classifier)
    """
    
    class_labels= np.unique(dataset['target'])
    binary_problem= (len(class_labels) == 2)
    
    dataset_orig_target= dataset['target'].copy()
    if binary_problem:
        # _logger.info("The problem is binary")
        if sum(dataset['target'] == class_labels[0]) < sum(dataset['target'] == class_labels[1]):
            min_label= class_labels[0]
            maj_label= class_labels[1]
        else:
            min_label= class_labels[0]
            maj_label= class_labels[1]
        
        min_ind= np.where(dataset['target'] == min_label)[0]
        maj_ind= np.where(dataset['target'] == maj_label)[0]
        np.put(dataset['target'], min_ind, 1)
        np.put(dataset['target'], maj_ind, 0)
    else:
        # _logger.info("The problem is not binary")
        label_indices= {}
        for c in class_labels:
            label_indices[c]= np.where(dataset['target'] == c)[0]
        mapping= {}
        for i, c in enumerate(class_labels):
            np.put(dataset['target'], label_indices[c], i)
            mapping[i]= c
    
    runtimes= []
    all_preds, all_tests= [], []
    
    for train, test in validator.split(dataset['data'], dataset['target']):
        # _logger.info("Executing fold")
        X_train, y_train= dataset['data'][train], dataset['target'][train]
        X_test, y_test= dataset['data'][test], dataset['target'][test]
        
        begin= time.time()
        X_samp, y_samp= sampler.sample(X_train, y_train)
        runtimes.append(time.time() - begin)
        
        X_samp_trans= scaler.fit_transform(X_samp)
        nonzero_var_idx= np.where(scaler.var_ > 1e-8)[0]
        X_test_trans= scaler.transform(X_test)
        
        all_tests.append(y_test)
        
        classifier.fit(X_samp_trans[:,nonzero_var_idx], y_samp)
        all_preds.append(classifier.predict_proba(X_test_trans[:,nonzero_var_idx]))
    
    if len(all_tests) > 0:
        all_preds= np.vstack(all_preds)
        all_tests= np.hstack(all_tests)
    
    dataset['target']= dataset_orig_target
    
    # _logger.info("Computing the results")
    
    results= {}
    results['runtime']= np.mean(runtimes)
    results['sampler']= sampler.__class__.__name__
    results['classifier']= classifier.__class__.__name__
    results['sampler_parameters']= str(sampler.get_params())
    results['classifier_parameters']= str(classifier.get_params())
    results['db_size']= len(dataset['data'])
    results['db_n_attr']= len(dataset['data'][0])
    results['db_n_classes']= len(class_labels)
    
    if binary_problem:
        results['imbalance_ratio']= sum(dataset['target'] == maj_label)/sum(dataset['target'] == min_label)
        all_pred_labels= np.apply_along_axis(lambda x: np.argmax(x), 1, all_preds)
    
        results['tp']= np.sum(np.logical_and(np.equal(all_tests, all_pred_labels), (all_tests == 1)))
        results['tn']= np.sum(np.logical_and(np.equal(all_tests, all_pred_labels), (all_tests == 0)))
        results['fp']= np.sum(np.logical_and(np.logical_not(np.equal(all_tests, all_pred_labels)), (all_tests == 0)))
        results['fn']= np.sum(np.logical_and(np.logical_not(np.equal(all_tests, all_pred_labels)), (all_tests == 1)))
        results['p']= results['tp'] + results['fn']
        results['n']= results['fp'] + results['tn']
        results['acc']= (results['tp'] + results['tn'])/(results['p'] + results['n'])
        results['sens']= results['tp']/results['p']
        results['spec']= results['tn']/results['n']
        results['ppv']= results['tp']/(results['tp'] + results['fp'])
        results['npv']= results['tn']/(results['tn'] + results['fn'])
        results['fpr']= 1.0 - results['spec']
        results['fdr']= 1.0 - results['ppv']
        results['fnr']= 1.0 - results['sens']
        results['bacc']= (results['tp']/results['p'] + results['tn']/results['n'])/2.0
        results['gacc']= np.sqrt(results['tp']/results['p']*results['tn']/results['n'])
        results['f1']= 2*results['tp']/(2*results['tp'] + results['fp'] + results['fn'])
        results['mcc']= (results['tp']*results['tn'] - results['fp']*results['fn'])/np.sqrt((results['tp'] + results['fp'])*(results['tp'] + results['fn'])*(results['tn'] + results['fp'])*(results['tn'] + results['fn']))
        results['l']= (results['p'] + results['n'])*np.log(results['p'] + results['n'])
        results['ltp']= results['tp']*np.log(results['tp']/((results['tp'] + results['fp'])*(results['tp'] + results['fn'])))
        results['lfp']= results['fp']*np.log(results['fp']/((results['fp'] + results['tp'])*(results['fp'] + results['tn'])))
        results['lfn']= results['fn']*np.log(results['fn']/((results['fn'] + results['tp'])*(results['fn'] + results['tn'])))
        results['ltn']= results['tn']*np.log(results['tn']/((results['tn'] + results['fp'])*(results['tn'] + results['fn'])))
        results['lp']= results['p']*np.log(results['p']/(results['p'] + results['n']))
        results['ln']= results['n']*np.log(results['n']/(results['p'] + results['n']))
        results['uc']= (results['l'] + results['ltp'] + results['lfp'] + results['lfn'] + results['ltn'])/(results['l'] + results['lp'] + results['ln'])
        results['informedness']= results['sens'] + results['spec'] - 1.0
        results['markedness']= results['ppv'] + results['npv'] - 1.0
        results['log_loss']= log_loss(all_tests, all_preds)
        results['auc']= roc_auc_score(all_tests, all_preds[:,1])
        test_labels, preds= zip(*sorted(zip(all_tests, all_preds[:,1]), key= lambda x: -x[1]))
        test_labels= np.array(test_labels)
        th= int(0.2*len(test_labels))
        results['p_top20']= np.sum(test_labels[:th] == 1)/th
        results['brier']= np.mean((all_preds[:,1] - all_tests)**2)
    else:
        all_pred_labels= np.apply_along_axis(lambda x: np.argmax(x), 1, all_preds)
        
        results['acc']= accuracy_score(all_tests, all_pred_labels)
        results['confusion_matrix']= confusion_matrix(all_tests, all_pred_labels)
        results['gacc']= gmean(np.diagonal(results['confusion_matrix'])/np.sum(results['confusion_matrix'], axis= 0))
        results['class_label_mapping']= mapping
        print(results['confusion_matrix'])

    return pd.DataFrame({'value': list(results.values())}, index= results.keys())