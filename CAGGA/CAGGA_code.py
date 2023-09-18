# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:24:20 2022
misclassification data -> cache
@author: xionglin
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
from unit import HIC
accuracy_list = np.array([])
hic = Hic()

@profile(precision=10,stream=open("LED_ARF.log", "w+"))
def test_train(ep,Mclf,X,y):
    for clf,clf_label in Mclf:
        Comp_start = time.time();y_pred = clf.predict(X);clf_a = accuracy_score(y, y_pred);clf.partial_fit(X, y,[1,0]);Time.append(time.time() - Comp_start);Acc.append(clf_a)
        # print(clf_label,"=",clf_a)
    #print(ep,end=".")
    

class CAGGA(object):
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
        """ add:HDDM_test 2022-8-2"""
        new_ghtc_acc = self.global_HTC.score(X_train,y_train)
        # print("global_clf_acc",len(accuracy_list))
        if hic(new_ghtc_acc,accuracy_list):
        # if change_detect(new_ghtc_acc,accuracy_list):
            print("detected warning...")
            ht_b = HoeffdingTreeClassifier()
            ht_b.fit(X_train,y_train)
            new_ht_acc = ht_b.score(X_train,y_train)
            self.Classifiers.append(ht_b)
            self.theta = np.append(self.theta,new_ht_acc)
        else:
            self.Classifiers[0].partial_fit(X_train, y_train)
            self.theta[0] = new_ghtc_acc

    # def train_Classifiers(self,X_train,y_train):
    #     ht_b = HoeffdingTreeClassifier()
    #     ht_b.fit(X_train,y_train)
    #     self.Classifiers.append(ht_b)
    #     self.theta = np.append(self.theta,1/mean_squared_error(ht_b.predict(X_train),y_train))
            
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
            self.global_HTC.fit(X,y)
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
                
            # min_w, max_w = min(w), max(w)
            # w = [(x-min_w)/(max_w-min_w+0.0001) for x in w]
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

def runOpemMLExp(sid):
    # try:
    # datalist = openml.datasets.list_datasets(output_format="dataframe")
    # datalist = datalist[(100000<datalist['NumberOfInstances'])&(datalist['NumberOfInstances']<1000000)].did
    
    stream = getOpenMLData( sid )
    print(stream)
    Mclf = init_model()
    global Acc
    Acc = [0.5]
    global Time
    Time = [] 
    num_tasks = 1
    num_row,n_ = stream.X.shape
    num_samples = int(num_row / num_tasks/100)
    print("instance={},features={} of dataset".format(num_row,n_))
    # num_samples = 1000
    model = CAGGA(stream,num_tasks,num_samples)
    for ep in range(num_tasks):
        X, y = stream.next_sample(num_samples)
        # X = X.astype(float)
        # X = preprocessing.scale(X,axis=0,with_std=True)
        y = y%2  #mutilabel --> binary label
        model.bulitModel(ep,X,y)
        test_train(ep,Mclf,X,y)  #vs other learner use test then train
    
    # #accuracy to csv
    # name=["GCUW"]+[x2 for (x1,x2) in Mclf]
    # Acc_mat = np.array(Acc).reshape(-1,len(name))
    # test=pd.DataFrame(columns=name,data=Acc_mat)
    # test = test.drop([0],axis=0)
    # # test.to_csv('{0}_A.csv'.format(stream.filename))
    # test.to_csv('{0}.{1}_A.csv'.format(stream.filename,sid))
    # print("\n",test.mean())
    # #time to csv
    # time_mat = np.array(Time).reshape(-1,len(name))
    # pd.DataFrame(columns=name,data=time_mat).to_csv('{0}.{1}_C.csv'.format(stream.filename,sid))
    # # if test.mean()[0]>=test.mean()[8] and test.mean()[0]>=test.mean()[3] and test.mean()[0]>=test.mean()[4] and test.mean()[0]>=test.mean()[5]:
    #     # saveid(stream.filename,sid+"\n")
    # # except:
    # #     print("error")
    
def expRunDataStream(stream):
    print(stream)
    Mclf = init_model()
    global Acc
    Acc = [0.5]
    global Time
    Time = [] 
    
    num_tasks = 10   #100
    num_samples = 500   #1000

    model = CAGGA(stream,num_tasks,num_samples)
    for ep in range(model.num_tasks):
        X, y = stream.next_sample(num_samples)
        # X = X.astype(float)
        # X = preprocessing.scale(X,axis=0,with_std=True)
        y = y%2  #mutilabel --> binary label
        #model.bulitModel(ep,X,y)
        test_train(ep,Mclf,X,y)  #vs other learner use test then train
    
    # #accuracy to csv
    # name=["GCUW"]+[x2 for (x1,x2) in Mclf]
    # Acc_mat = np.array(Acc).reshape(-1,len(name))
    # test=pd.DataFrame(columns=name,data=Acc_mat)
    # test = test.drop([0],axis=0)
    # # test.to_csv('{0}_A.csv'.format(stream.filename))
    # test.to_csv('{0}._A.csv'.format(stream.name))
    # print("\n",test.mean())
    # #time to csv
    # time_mat = np.array(Time).reshape(-1,len(name))
    # pd.DataFrame(columns=name,data=time_mat).to_csv('{0}_C.csv'.format(stream.name))

# def Times_Exp(stream):
#     for i in range(100):
#         try:
#             expRunDataStream(stream)
#             # expRunStrStream(stream)
            
#         except:
#             continue
#         else:
#             break

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
    num_tasks = 10
    num_samples = stream.chunk_size
    
    model = CAGGA(stream,num_tasks,num_samples)
    for ep in range(model.num_tasks):
        X, y = stream.get_chunk()
        # model.global_HTC.fit(X,y)
        y = y%2  #mutilabel --> binary label
        #model.bulitModel(ep,X,y)
        test_train(ep,Mclf,X,y)  #vs other learner use test then train
    
    # #accuracy to csv
    # name=["GCUW"]+[x2 for (x1,x2) in Mclf]
    # Acc_mat = np.array(Acc).reshape(-1,len(name))
    # test=pd.DataFrame(columns=name,data=Acc_mat)
    # test = test.drop([0],axis=0)
    # # test.to_csv('{0}_A.csv'.format(stream.filename))
    # test.to_csv('{0}._A.csv'.format(stream))
    # print("\n",test.mean())
    # #time to csv
    # time_mat = np.array(Time).reshape(-1,len(name))
    # pd.DataFrame(columns=name,data=time_mat).to_csv('{0}_C.csv'.format(stream))

def saveid(sid):
    f=open('fsid.txt','a')
    f.write("{},".format(sid))
    f.close()

    
def runOpenML_Exp():
    # datalist = [1204, 44092, 43975, 42729, 44058, 44129, 44050, 44018]
    # datalist = [44092, 43975]
    datalist = [727,881,901, 1201]
    # # datalist = [44032, 44094, 42733, 44147, 881,901, 44049, 44050, 44058, 296, 1201,1203, 1204, 44092, 43975, 43987, 43988, 727, 43992,44032, 43996, 44129, 42729, 44137, 42731, 44013, 42733, 881, 44018, 44147, 44019, 44026, 44031]
    # pool = multiprocessing.Pool()
    # # inputs = [stream for stream in streams]
    # inputs = [data for data in datalist]
    # outputs = pool.map(runOpemMLExp,inputs)
    for dataid in datalist:
        runOpemMLExp(dataid)

def runStrlearn_Exp():
    streams = getStrlearnstream()
    # pool = multiprocessing.Pool()
    # inputs = [stream for stream in streams]
    # outputs = pool.map(expRunStrStream,inputs)
    for stream in streams:
        expRunStrStream(stream)

def runMultiflow_Exp():
    streams = getmultiflowStreams()
    # pool = multiprocessing.Pool()
    # inputs = [stream for stream in streams]
    # outputs = pool.map(expRunDataStream,inputs)
    for stream in streams:
        expRunDataStream(stream)
    
def runfilestream():
    filename = r'C:\Users\xionglin\Desktop\xl\data\BNG(breastTumor).csv'
    stream = FileStream(filename)
    print(stream)
    Mclf = init_model()
    global Acc
    Acc = [0.5]
    global Time
    Time = [] 
    num_tasks = 10
    num_row,n_ = stream.X.shape
    num_samples = int(num_row / num_tasks)
    num_samples = num_samples if num_samples<=500 else 500
    print("instance={},features={} of dataset".format(num_row,n_))
    model = CAGGA(stream,num_tasks,num_samples)
    for ep in range(num_tasks):
        X, y = stream.next_sample(num_samples)
        # X = X.astype(float)
        # X = preprocessing.scale(X,axis=0,with_std=True)
        y = y%2  #mutilabel --> binary label
        #model.bulitModel(ep,X,y)
        test_train(ep,Mclf,X,y)  #vs other learner use test then train
        
    # name=["GCUW"]+[x2 for (x1,x2) in Mclf]
    # Acc_mat = np.array(Acc).reshape(-1,len(name))
    # test=pd.DataFrame(columns=name,data=Acc_mat)
    # test = test.drop([0],axis=0)
    # # test.to_csv('{0}_A.csv'.format(stream.filename))
    # test.to_csv('{0}_A.csv'.format(stream.filename))
    # # print("\n",test.mean())
    # #time to csv
    # time_mat = np.array(Time).reshape(-1,len(name))
    # pd.DataFrame(columns=name,data=time_mat).to_csv('{0}_C.csv'.format(stream.filename))
    
if __name__ == '__main__':
    # runOpenML_Exp()
    runStrlearn_Exp()
    #runMultiflow_Exp()
    #runfilestream()
    print("\a")

#diabetic_data   census  NonSkin
#2dplanes fried   mv BNG(breastTumor)