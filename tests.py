import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
import os 
import h5py
import utilities

a = 'k_ya'

if '_' in a:
    a = a.replace('_', '')
    print (a)
