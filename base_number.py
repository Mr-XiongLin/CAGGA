# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:09:42 2023

@author: xionglin
'
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import warnings
warnings.filterwarnings("ignore", category=Warning)

from sklearn import ensemble
from openml import tasks, runs
import numpy as np
import pandas as pd
import time,copy,math
from sklearn.model_selection import train_test_split
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.drift_detection import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone,BaseEstimator,TransformerMixin
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import *
from skmultiflow.meta import *
from skmultiflow.lazy import *
from skmultiflow.trees import *
from sklearn import preprocessing
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from sklearn.utils import shuffle
import strlearn
from sklearn.datasets import fetch_openml
from strlearn.streams import StreamGenerator
from skmultiflow.data.file_stream import FileStream
from memory_profiler import profile
import strlearn as sl
global accuracy_list
accuracy_list = np.array([])
hddm_a = HDDM_A()

def init_model():
    ht = HoeffdingTreeClassifier()#2010
    hat = HoeffdingAdaptiveTreeClassifier()#
    # eft = ExtremelyFastDecisionTreeClassifier()#
    # samknn = SAMKNNClassifier(n_neighbors=5, weighting='distance', max_window_size=1000,stm_size_option='maxACCApprox', use_ltm=False)#
    # lch = LabelCombinationHoeffdingTreeClassifier(n_labels=2)#
    AWE = AccuracyWeightedEnsembleClassifier()#2003
    DWM = DynamicWeightedMajorityClassifier(n_estimators=50)#2007
    ARF = AdaptiveRandomForestClassifier()#2017
    oBoost = OnlineBoosting()#2016
    lpp = LearnPPClassifier()#2002
    learn_pp_nse = LearnPPNSEClassifier()#2011
    online_adac2 = OnlineAdaC2Classifier()#2016
    # online_smote = OnlineSMOTEBaggingClassifier()#2016
    Batch_I = BatchIncrementalClassifier()#
    aee = AdditiveExpertEnsembleClassifier(n_estimators=100)#2005
    leverbagging = LeveragingBaggingClassifier()#2010
    OzaBagging = OzaBaggingClassifier()#2005
    SRP = StreamingRandomPatchesClassifier(base_estimator=KNNClassifier(n_neighbors=5), random_state=1,n_estimators=3)#2019
    pcc = ProbabilisticClassifierChain()#2011
    # cc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1))#2009
    mcc = MonteCarloClassifierChain()#2011,
    Mclf = [(aee,"AEE")] 
    # Mclf = [(DWM,'DWM'),(Batch_I,"BAI"),(ARF,'ARF'),(aee,'AEE'),(leverbagging,"LevBagg"),(OzaBagging,"OzaBagg"),(SRP,"SRP")]#,(oBoost,"oBoost")
    return Mclf

# @profile(precision=10,stream=open("LED_ARF.log", "w+"))
def test_train(ep,Mclf,X,y):
    for clf,clf_label in Mclf:
        print(clf_label,clf.n_estimators)
        Comp_start = time.time();y_pred = clf.predict(X);clf_a = accuracy_score(y, y_pred);clf.partial_fit(X, y,[1,0]);Time.append(time.time() - Comp_start);Acc.append(clf_a)
        # print(clf_label,"=",clf_a)
    #print(ep,end=".")
    

class DSGA_MAML(object):
    def __init__(self,stream,num_tasks,num_samples):
        #initialize number of tasks i.e number of tasks we need in each batch of tasks
        self.num_tasks = num_tasks
        #number of samples i.e number of shots  -number of data points (k) we need to have in each task
        self.num_samples = num_samples
        #number of epochs i.e training iterations
        self.epochs = 100
        #hyperparameter for the inner loop (inner gradient update)
        self.alpha = .01
        #hyperparameter for the outer loop (outer gradient update) i.e meta optimization
        self.beta = .01
        #classifier collection
        self.theta = [0.6]
        self.global_HTC = HoeffdingTreeClassifier()
        self.Classifiers = [self.global_HTC]
    
    def sigmoid(self,a):
        return 1.0 / (1 + np.exp(-a))

    
    def train_Classifiers(self,X_train,y_train):
        """ add:HDDM_test 2023-2-13"""
        new_ghtc_acc = self.Classifiers[0].score(X_train,y_train)
        hddm_a.add_element(1-new_ghtc_acc)
        if hddm_a.detected_warning_zone():
            print("detected warning...")
            ht_b = HoeffdingTreeClassifier()
            ht_b.partial_fit(X_train,y_train)
            new_ht_acc = ht_b.score(X_train,y_train)
            self.Classifiers.append(ht_b)
            self.theta = np.append(self.theta,new_ht_acc)
        else:
            self.Classifiers[0].partial_fit(X_train, y_train)
            self.theta[0] = new_ghtc_acc

    def classifiers_out(self,X_input):
        probas = np.asarray([clf.predict_proba(X_input)[:,-1] for clf in self.Classifiers])
        return probas.reshape(X_input.shape[0],-1)
    
    #@profile (precision=10,stream=open("Stream2G_GC.log", "w+"))
    def bulitModel(self,ep,X,y):
        if ep > 0:
            y_my_pred = (self.predict(X)>=0.5).astype(int)
            DSGA_accuracy = accuracy_score(y,y_my_pred)
            Acc.append(DSGA_accuracy)
        else:
            self.global_HTC.partial_fit(X, y)
        #proposed algrithm train start,record the start time
        NA_start = time.time()
        # XTrain,XTest,YTrain,YTest = train_test_split(X,y,random_state=11,test_size=0.2)
        XTrain,XTest,YTrain,YTest = X,X,y,y
        # self.train_Classifiers(XTrain,YTrain)
        self.train_Classifiers(X,y)
        nn_input = self.classifiers_out(XTrain)
        nn_meta_input = self.classifiers_out(XTest)
        
        self.theta = self.theta/max(self.theta)  #normalization
        MC_len = len(self.Classifiers)
        self.g = np.array([])
        for e in range(self.epochs):
            self.theta_ = np.array([])
            #for storing gradient updates
            #for each base classfier, calc the weights gradient
            a = nn_input*self.theta
            YHat = self.sigmoid(a)
            gradient = np.dot(nn_input.T,(YHat-YTrain.reshape(-1,1))).sum(axis=0) /(self.num_samples*1)
            self.theta_ = np.append(self.theta_,self.theta-self.alpha*gradient)
            self.g = np.append(self.g,self.theta-self.theta_)
            normalization_factor = 0.000000000000000001
            for i in range(MC_len):
                for j in range(MC_len):      
                    normalization_factor += np.abs(np.dot(self.g[i].T, self.g[j]))
            w = np.zeros(MC_len)
            for i in range(MC_len):
                for j in range(MC_len):
                    w[i] += np.dot(self.g[i].T, self.g[j])
                w[i] = w[i] / normalization_factor

            #initialize meta gradients
            weighted_gradient = np.zeros(MC_len)
            meta_a = np.matmul(nn_meta_input,self.theta_)
            YPred = self.sigmoid(meta_a)
            #compute meta gradients
            meta_gradient = np.dot(nn_meta_input.T,(YPred-YTest)) / (self.num_samples*1)
            weighted_gradient += np.sum(w*meta_gradient)
            self.theta = self.theta + self.beta*weighted_gradient/MC_len
        # print(self.theta)
        Time.append(time.time() - NA_start)

    def predict(self,X_input):
        """predict XTest label"""
        Classifiers_weight = self.theta/self.theta.sum()
        probas = np.asarray([clf.predict_proba(X_input)[:,-1]*w for (clf,w) in zip(self.Classifiers,Classifiers_weight)])
        # probas = np.asarray([clf.predict_proba(X_input)[:,-1] for clf in self.Classifiers])
        return probas.sum(axis=0)

    

def expRunStrStream(stream):
    print(stream)
    Mclf = init_model()
    global Acc
    Acc = [0.5]
    global Time
    global global_clf_Acc
    global_clf_Acc= []
    Time = [] 
    # num_tasks = stream.n_chunks
    num_tasks = stream.n_chunks
    num_samples = stream.chunk_size
    
    model = DSGA_MAML(stream,num_tasks,num_samples)
    for ep in range(100):
        print("\niter:",ep)
        X, y = stream.get_chunk()
        model.global_HTC.partial_fit(X, y)
        y = y%2  #mutilabel --> binary label
        model.bulitModel(ep,X,y)
        print("my",len(model.Classifiers))
        test_train(ep,Mclf,X,y)  #vs other learner use test then train
        
    #accuracy to csv
    name=["GCUW"]+[x2 for (x1,x2) in Mclf]
    Acc_mat = np.array(Acc).reshape(-1,len(name))
    test=pd.DataFrame(columns=name,data=Acc_mat)
    test = test.drop([0],axis=0)
    # test.to_csv('{0}_A.csv'.format(stream.filename))
    test.to_csv('{0}._A.csv'.format(stream))
    print("\n",test.mean())
    #time to csv
    time_mat = np.array(Time).reshape(-1,len(name))
    pd.DataFrame(columns=name,data=time_mat).to_csv('{0}_C.csv'.format(stream))
    
if __name__ == '__main__':
    stream = StreamGenerator(n_chunks=100,chunk_size=500, n_drifts = 10)
    expRunStrStream(stream)
    print("\a")
