# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:57:05 2023
#https://huiwenn.github.io/spectral-clustering?fbclid=IwAR2SdwjEtGCLJlfDhOUb4d5mYqGhJL-ahNvU-LdabBBzWsHe21jSNtqYhZA
@author: gmostafa
"""
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist
import cv2
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt

import scipy.io as sio

def A_local(X):
    dim = X.shape[0]
    dist_ = pdist(X)
    pd = np.zeros([dim, dim])
    dist = iter(dist_)
    for i in range(dim):
        for j in range(i+1, dim):  
            d = next(dist)
            pd[i,j] = d
            pd[j,i] = d
        
    #calculate local sigma
    sigmas = np.zeros(dim)
    for i in tqdm(range(len(pd))):
        sigmas[i] = sorted(pd[i])[7]

    A = np.zeros([dim, dim])
    dist = iter(dist_)
    for i in tqdm(range(dim)):
        for j in range(i+1, dim):  
            d = np.exp(-1*next(dist)**2/(sigmas[i]*sigmas[j]))
            #print(d)
            A[i,j] = d
            A[j,i] = d
    return A



im_ = cv2.imread("C:/Users/gmostafa/Downloads/Dr Howle/BC480.jpg", cv2.IMREAD_COLOR)
im = cv2.cvtColor(im_, cv2.COLOR_BGR2GRAY)
plt.imshow(im, cmap = 'gray') 

sz = im.itemsize
h,w = im.shape
bh,bw = 2,2 #block height and width
shape = (int(h/bh), int(w/bw), bh, bw)
strides = sz*np.array([w*bh,bw,w,1])
print(shape, strides)

patches = np.lib.stride_tricks.as_strided(im, shape=shape, strides=strides)


a,b,c,d = patches.shape
X = patches.reshape([a*b, c*d])
A = A_local(X)
D_half = np.linalg.fractional_matrix_power(np.sum(A, axis=0) * np.eye(X.shape[0]), -0.5)
L = np.matmul(np.matmul(D_half, A), D_half)
sio.savemat(L)

def whatever():
    sio.savemat('L.mat',{'data': L})
    return
whatever()

# eigval, eigvec = np.linalg.eigh(L)


# def whatever():
#     #L = np.matmul(np.matmul(D_half, A), D_half)
#     evals, evecs = np.linalg.eig(L)
#     #Assume that the eigenvalues are ordered from large to small and that the
#     #eigenvectors are ordered accordingly.
#     sio.savemat('eigval2.mat', {'data': evals[1]})

#     sio.savemat('eigvec2.mat', {'data': evecs[:,1]})
#     return
# whatever()

# def eig2pic_(eig):
#     arr_blocks = eig.reshape([a, b]) #a, b comes from code block above
#     img = np.array([np.hstack(bl) for bl in arr_blocks])
#     img = np.vstack(img)
#     plt.imshow(img, cmap = 'gray') 

#eig2pic_(eigvec_cat[:,-1])



