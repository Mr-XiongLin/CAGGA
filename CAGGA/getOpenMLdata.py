# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:31:28 2023

@author: xionglin


"""
from skmultiflow.data.file_stream import FileStream
from sklearn.datasets import fetch_openml
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def getOpenMLData(data_idnumber):
    enc = preprocessing.OrdinalEncoder()
    print("data_idnumber:",data_idnumber)
    D = fetch_openml(data_id= data_idnumber )
    X = D.data
    y = D.target
    X["y"]=np.array(y).reshape(-1,1)
    X = pd.DataFrame(X)
    if X.isnull().values.any():
        X = X.fillna(0,inplace=True)
    X = enc.fit(X).transform(X)
    X = shuffle(X)
    filename = r'C:\Users\xionglin\Desktop\xl\data\{}.csv'.format(D.details['name'])
    # pd.DataFrame(X).to_csv(filename,header=None)
    return FileStream(filename)