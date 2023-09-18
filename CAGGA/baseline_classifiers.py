# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:36:48 2023

@author: xionglin
"""

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone,BaseEstimator,TransformerMixin
from skmultiflow.data import *
from skmultiflow.meta import *
from skmultiflow.lazy import *
from skmultiflow.trees import *

def init_model():
    ht = HoeffdingTreeClassifier()#2010
    hat = HoeffdingAdaptiveTreeClassifier()#
    # eft = ExtremelyFastDecisionTreeClassifier()#
    # samknn = SAMKNNClassifier(n_neighbors=5, weighting='distance', max_window_size=1000,stm_size_option='maxACCApprox', use_ltm=False)#
    # lch = LabelCombinationHoeffdingTreeClassifier(n_labels=2)#
    AWE = AccuracyWeightedEnsembleClassifier()#2003
    DWM = DynamicWeightedMajorityClassifier()#2007
    ARF = AdaptiveRandomForestClassifier()#2017
    oBoost = OnlineBoostingClassifier()#2016
    lpp = LearnPPClassifier()#2002
    learn_pp_nse = LearnPPNSEClassifier()#2011
    online_adac2 = OnlineAdaC2Classifier()#2016
    # online_smote = OnlineSMOTEBaggingClassifier()#2016
    Batch_I = BatchIncrementalClassifier()#
    aee = AdditiveExpertEnsembleClassifier()#2005
    leverbagging = LeveragingBaggingClassifier()#2010
    OzaBagging = OzaBaggingClassifier()#2005
    SRP = StreamingRandomPatchesClassifier(base_estimator=KNNClassifier(n_neighbors=5), random_state=1,n_estimators=3)#2019
    pcc = ProbabilisticClassifierChain()#2011
    # cc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1))#2009
    mcc = MonteCarloClassifierChain()#2011,
    Mclf = [(ARF,'ARF')] 
    # Mclf = [(DWM,'DWM'),(Batch_I,"BAI"),(ARF,'ARF'),(aee,'AEE'),(leverbagging,"LevBagg"),(OzaBagging,"OzaBagg"),(SRP,"SRP"),(oBoost,"oBoost")]#
    return Mclf

def SOTA_model_strlearn():
    Mclf=[(sl.ensembles.AUE(),"AUE"),(sl.ensembles.AWE(),"AWE"),(sl.ensembles.CDS,"CDS"),(sl.ensembles.DWM(),"DWM"),(sl.ensembles.KMC(),"KMC"),\
          (sl.ensembles.KUE(),"KUE"),(sl.ensembles.NIE(),"NIE"),(sl.ensembles.OOB(),"OOB"),(sl.ensembles.OUSE(),"OUSE"),(sl.ensembles.OnlineBagging(),"oBagg"),\
          (sl.ensembles.REA(),"REA"),(sl.ensembles.SEA(),"SEA"),(sl.ensembles.StreamingEnsemble(),"SE"),(sl.ensembles.UOB(),"UOB"),(sl.ensembles.WAE(),"WAE")]
    return Mclf