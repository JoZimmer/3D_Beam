import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
import os 
import h5py
import utilities

#print (utilities.eigenmodes_target_2D_3_elems)
l= np.array([[1,-1],[1e-10,-3]])

od = l <0
print (l[od])