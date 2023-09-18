# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 11:31:21 2023

@author: xionglin
"""

# Imports
import time
import numpy as np
from skmultiflow.bayes import NaiveBayes
from sklearn.metrics import accuracy_score
from skmultiflow.drift_detection.adwin import ADWIN
adwin = ADWIN(delta=2)

from skmultiflow.drift_detection import DDM
ddm = DDM()#min_num_instances=10, warning_level=1.2, out_control_level=3.0

from skmultiflow.drift_detection.eddm import EDDM
eddm = EDDM()

from skmultiflow.drift_detection.hddm_a import HDDM_A
hddm_a = HDDM_A(drift_confidence=0.001)  #1-2

from skmultiflow.drift_detection import KSWIN
kswin = KSWIN(alpha=0.01)

from skmultiflow.drift_detection import PageHinkley
ph = PageHinkley()#min_instances=10, delta=0.5, threshold=20, alpha=0.8

exp_inter = 10
from strlearn.streams import StreamGenerator
stream = StreamGenerator(n_chunks=10,chunk_size=5000,n_drifts=2000)

from skmultiflow.trees import HoeffdingTreeClassifier
ht = NaiveBayes()#

Acc=[]
Time=[]
def test_train(ep,Mclf,X,y):
    clf = Mclf
    Comp_start = time.time();
    y_pred = clf.predict(X);
    clf_a = accuracy_score(y, y_pred);
    clf.partial_fit(X, y,[1,0]);
    Time.append(time.time() - Comp_start);
    Acc.append(clf_a)
        # print(clf_label,"=",clf_a)
    #print(ep,end=".")


X, y = stream.get_chunk()
y = y%2
for i in range(len(X)):
    test_train(i,ht,X[i],y[i])

Acc = np.array(Acc)
data_stream = Acc
# # Simulating a data stream as a normal distribution of 1's and 0's
# data_stream = np.random.randint(2, size=10000)/10.0
# # Changing the data concept from index 999 to 2000
# for i in range(999, 2000):
#     data_stream[i] = np.random.randint(20, high=30)/100.0
# for i in range(2000, 2999):
#     data_stream[i] = np.random.randint(30, high=40)/100.0
# for i in range(3000, 3999):
#     data_stream[i] = np.random.randint(40, high=50)/100.0
# for i in range(5000, 5999):
#     data_stream[i] = np.random.randint(50, high=60)/100.0
# for i in range(6000, 6999):
#     data_stream[i] = np.random.randint(60, high=70)/100.0
    
# for i in range(7000, 7999):
#     data_stream[i] = np.random.randint(70, high=80)/100.0
    
# for i in range(8000, 8999):
#     data_stream[i] = np.random.randint(80, high=90)/100.0
    
# for i in range(9000, 10000):
#     data_stream[i] = np.random.randint(90, high=100)/100.0


# adwin_record = np.zeros((exp_inter,))
# for i in range(exp_inter):
#     adwin.add_element(data_stream[i])
#     if adwin.detected_change():
#         print('adwin Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
#         print(data_stream[i])
#         adwin_record[i] = data_stream[i]
#     else:
#         adwin_record[i] = 0
        

ddm_record = np.zeros((exp_inter,))
for i in range(exp_inter):
    ddm.add_element(data_stream[i])
    if ddm.detected_change():
        print('ddm Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
        print(data_stream[i])
        ddm_record[i] = data_stream[i]
    else:
        ddm_record[i] = 0
        
        
eddm_record = np.zeros((exp_inter,))
for i in range(exp_inter):
    eddm.add_element(data_stream[i])
    if eddm.detected_change():
        print('eddm Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
        print(data_stream[i])
        eddm_record[i] = data_stream[i]
    else:
        eddm_record[i] = 0
        
        
        
hddm_a_record = np.zeros((exp_inter,))
for i in range(exp_inter):
    hddm_a.add_element(data_stream[i])
    if hddm_a.detected_change():
        print('hddm_a Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
        print(data_stream[i])
        hddm_a_record[i] = data_stream[i]
    else:
        hddm_a_record[i] = 0
        
        
# kswin_record = np.zeros((exp_inter,))
# for i in range(exp_inter):
#     kswin.add_element(data_stream[i])
#     if kswin.detected_change():
#         print('kswin Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
#         print(data_stream[i])
#         kswin_record[i] = data_stream[i]
#     else:
#         kswin_record[i] = 0
        
        
ph_record = np.zeros((exp_inter,))
for i in range(exp_inter):
    ph.add_element(data_stream[i])
    if ph.detected_change():
        print('ph Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
        print(data_stream[i])
        ph_record[i] = data_stream[i]
    else:
        ph_record[i] = 0