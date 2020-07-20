#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:53:09 2020

@author: phil

# Randomized SVD code taken from http://databookuw.com 
"""
import numpy as np


def rSVD(X,r,q,p):
    """
    X snapshotmatrix
    r truncation rank
    q number of power iterations
    p oversampling
    """
    # Step 1: sample column space of X with P matrix
    ny = X.shape[1]
    P = np.random.randn(ny,r+p)
    Z = X @ P
    for k in range (q):
        Z = X @ (X.T @ Z)
        
    Q, R = np.linalg.qr(Z, mode='reduced')
    
    # Step2: compute SVD on projected Y = Q.T@X
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices = 0)
    U = Q @ UY
    
    return U,S,VT