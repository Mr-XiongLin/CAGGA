# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 19:55:51 2022

@author: xionglin
"""

import Orange
# import matplotlib
# matplotlib.use('TkAgg')  # 不显示图则加上这两行
import matplotlib.pyplot as plt

names = ['DWM',"BAI",'ARF','AEE',"LevBagg","OzaBagg","SRP","oBoost","CAGGA"]
#avranks = [5.45, 3.54, 2.18, 5.95, 5.72, 4.95, 4.22, 8.09, 1.50]
avranks = [3.40, 1.00, 5.36, 3.59, 7.63, 7.00, 5.81, 9.00, 2.09]
#avranks = [2.00, 3.59, 8.04, 2.90, 7.45, 7.81, 6.63, 1.68, 4.45]
datasets_num = 16
CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=9, textspace=2, reverse=False)
print(CD)
plt.show()
