<!--
 * @Author: Zhou Hao
 * @Date: 2022-04-07 18:04:04
 * @LastEditors: Zhou Hao
 * @LastEditTime: 2022-04-07 18:15:56
 * @Description: file content
 * @E-mail: 2294776770@qq.com
-->



# SW: A Weighted Space Division Framework for Imbalanced Problems with Label Noise

* Abstract—Imbalanced data learning is a ubiquitous challenge in data mining and machine learning. The synthetic minority oversampling technique (SMOTE) and its variants have been proposed to address this problem. The process utilized by these variants is to emphasize the specific area or combine them with different noise filters; it introduces additional parameters that are difficult to optimize or that rely on specific noise filters. Furthermore, SMOTE-based methods randomly select the nearest neighbor samples and randomly perform interpolation to synthesize new samples without considering the impact of the chaotic degree of the sample space. In this work, a framework called SW that measures chaos of the sample space to weighted sampling is proposed. It is a general, robust and adaptive framework that can be combined with various oversampling algorithms but it does not work alone. In the SW framework, complete random forest (CRF) is introduced to divide the sample space and adaptively assign weights that are used to distinguish and filter noisy and outlier samples. When synthesizing a new sample, SW selects the neighboring samples of the seed samples(minority samples used to synthesize new samples ) and calculates the specific informed position with the derived weights, which brings the new sample closer to the safe area. Experimental results on 16 benchmark datasets and 8 classic classifiers with eight pairs of representative oversampling algorithms significantly demonstrate the effectiveness of the SW framework. The implementation of the proposed SW framework in the Python programming language is available at https://github.com/dream-lm/SW_framework.

* _smote_variants_v1.py: Post-processing algorithms.

* _smote_variants_v3.py: Post-processing algorithms combined with SW framework.

* all_smote_v3.py: Pre-processing processing algorithm combined with SW framework.

* crf_weight_api.py: The weighted interpolation function called by _smote_variants_v3.py.


* CRF.py: Code of Completely Random Forest.

* main_help.py: add_flip_noise() and load_data().

* main.py: Sample code for the SW framework.

* requirements: python environment.

# Requirements

### Minimal installation requirements (Python 3.7):

* Anaconda 3.
  
* Linux operating system or Windows operating system.

* Sklearn, numpy, pandas, imbalanced_learn.




### Installation requirements (Python 3):

* pip install -r requirements.txt


# Usage

* pip install -r requirements.txt.
* python main.py

# Doesn't work?

* Please contact Hao Zhou at 2294776770@qq.com
