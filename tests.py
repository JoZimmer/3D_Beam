import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
import os 
import h5py
import utilities
from  postprocess import Postprocess

d = np.asarray([[2,4],
                [1,2],
                [2,4],
                [1,2],])

f = np.zeros(10)

f[2::2] = d[:,1]

m = np.apply_along_axis(np.mean, 1, d)

print (f)