# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:35:32 2023

@author: xionglin
"""

from skmultiflow.data import *
from skmultiflow.meta import *
from skmultiflow.lazy import *
from skmultiflow.trees import *

def getmultiflowStreams():
    streams =[
            #SEAGenerator(classification_function = 2, random_state = 112,balance_classes = True),
            #ConceptDriftStream(stream=AGRAWALGenerator(balance_classes=False, classification_function=0, perturbation=0.0,random_state=112), drift_stream=AGRAWALGenerator(balance_classes=False, classification_function=2,perturbation=0.0, random_state=112), position=10000, width=1000, random_state=None, alpha=1.0),
            # SineGenerator(classification_function = 2, random_state = 112,balance_classes = False, has_noise = True),
            # STAGGERGenerator(classification_function = 2, random_state = 112,balance_classes = False),
            # WaveformGenerator(random_state=774, has_noise=True),
            #RandomTreeGenerator(tree_random_state=None, sample_random_state=None, n_classes=2, n_cat_features=5, n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=5, min_leaf_depth=3, fraction_leaves_per_level=0.15),
            # RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=4,n_features=10, n_centroids=50),
            # RandomRBFGeneratorDrift(model_random_state=99, sample_random_state = 50,n_classes = 2, n_features = 10, n_centroids = 2, change_speed=0.57,num_drift_centroids=10),
            # MultilabelGenerator(n_samples=100100, n_features=20, n_targets=1, n_labels=4, random_state=112),
            #MIXEDGenerator(classification_function = 1, random_state= 112,balance_classes = True),
            LEDGenerator(random_state = 112, noise_percentage = 0.18, has_noise= True),
            # LEDGeneratorDrift(random_state = 112, noise_percentage = 0.2,has_noise= True,n_drift_features=4),
            # HyperplaneGenerator(random_state=None, n_features=10, n_drift_features=2, mag_change=0.0, noise_percentage=0.30, sigma_percentage=0.1),
            # AnomalySineGenerator(n_samples=100100, n_anomalies=50000, contextual=False, n_contextual=35000, shift=4, noise=0.5, replace=True, random_state=112),
            # AGRAWALGenerator(classification_function=2, random_state=112, balance_classes=False, perturbation=0.2)
        ]
    return streams