import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
import os 
import h5py

yg = 2430

k = np.array([[ 1920.        ,     0.        ,  -960.        , -yg],
       [    0.        , 16000.        ,  yg,  4000.        ],
       [ -960.        ,  yg,   960.        ,  yg],
       [-yg,  4000.        ,  yg,  8000.        ]])

f = np.array([[0.],
       [0.],
       [-1.],
       [0.]])

m = np.array([[11,21,31,41],
              [10,20,30,40],
              [12,22,32,42]])

v = np.array([1.1,2.1,3,4])/100
w = np.array([1,2,3,4])/100
z = np.linalg.norm(v-w)
scaling = 10**-2

print (z)

