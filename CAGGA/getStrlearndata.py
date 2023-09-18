# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:33:55 2023

@author: xionglin
"""

from strlearn.streams import StreamGenerator

def getStrlearnstream():
    streams =[
        # StreamGenerator(n_chunks=100,chunk_size=500),
        #StreamGenerator(n_chunks=100,chunk_size=500,n_drifts=2), #2A
        StreamGenerator(n_chunks=100,chunk_size=500, n_drifts = 5),  #5A
        # StreamGenerator(n_chunks=100,chunk_size=500,n_drifts=2, concept_sigmoid_spacing=5),
        # StreamGenerator(n_chunks=100,chunk_size=500,n_drifts=2, concept_sigmoid_spacing=5, incremental=True), #2G
        # StreamGenerator(n_chunks=100,chunk_size=500,n_drifts=4, recurring=True),
        # StreamGenerator(n_chunks=100,chunk_size=500,weights=[0.2, 0.8]),
        # StreamGenerator(n_chunks=100,chunk_size=500,weights=(2, 5, 0.9)),
        # StreamGenerator(n_chunks=100,chunk_size=500,weights=(2, 5, 0.9), n_drifts=3, concept_sigmoid_spacing=5,recurring=True, incremental=True)
        ]
    return streams
