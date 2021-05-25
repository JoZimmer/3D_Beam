import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
import os 
import h5py
import utilities



eig_freq_cur =  0.53#0.42899971233201306
target_freq = 0.536

r = ((eig_freq_cur - target_freq)**2 *100)#/ target_freq**3)

print(72 * 24**3 /12 + 746496.0)
