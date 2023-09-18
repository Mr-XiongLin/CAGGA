
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 19:32:24 2022
memory cost
@author: Administrator
"""
import numpy as np
import pandas as pd
import time,copy,math
from sklearn.model_selection import train_test_split
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone,BaseEstimator,TransformerMixin
from skmultiflow.data import *
from skmultiflow.meta import *
from sklearn import preprocessing
from skmultiflow.bayes import NaiveBayes
from memory_profiler import profile
from strlearn.streams import StreamGenerator
def init_model():
    DWM = DynamicWeightedMajorityClassifier()#2007
    Batch_I = BatchIncrementalClassifier()#
    ARF = AdaptiveRandomForestClassifier()#2017
    aee = AdditiveExpertEnsembleClassifier()#2005
    leverbagging = LeveragingBaggingClassifier()#2010
    OzaBagging = OzaBaggingClassifier()#2005
    SRP = StreamingRandomPatchesClassifier(base_estimator=NaiveBayes(), random_state=1,n_estimators=3)#2019
    oBoost = OnlineBoostingClassifier()#2016
    # Mclf = [(aee,'AEE')] 
    # Mclf = [(DWM,'DWM'),(Batch_I,"Batch_I"),(ARF,'ARF'),(aee,'AEE'),(leverbagging,"leverBagg"),(OzaBagging,"OzaBagg"),(SRP,"SRP"),(oBoost,"oBoost")]#
    return Mclf

@profile(precision=10)
def my_func(X,y):
    ht = OnlineBoostingClassifier()#2016
    
    ht.partial_fit(X, y,[1,0])
    


if __name__ == '__main__':
    streams = getStrlearnstream()
    for stream in streams:
        X, y = stream.get_chunk()
        # X, y = stream.next_sample(1000)
        X = preprocessing.scale(X,axis=0,with_std=True)
        y = y%2
        my_func(X,y)