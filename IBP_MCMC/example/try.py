"""
Run IBP on the synthetic 'Cambridge Bars' dataset
"""
import sys
import time
import os, sys
import shutil
import numpy as NP
import scipy.io as SPIO
import math

######################## best-fit###################
def best_fit(Z0,rZ):
    dis,tk = Z0.shape
    rk = rZ.shape[1]
    b_Z = NP.zeros((dis,tk))
    for i in range(tk):
        print 'i=%d' %i
        best_index = 0
        best_dist = dis
        for j in range(rk):
            print 'j=%d' %j
            if  NP.count_nonzero(Z0[:,i]!=rZ[:,j]) < best_dist:
                print 'change'
                best_index = j
                best_dist = NP.count_nonzero(Z0[:,i]!=rZ[:,j])
        b_Z [:,i] = rZ [:,best_index]
    h_err = NP.count_nonzero(Z0!=b_Z)/float(Z0.shape[0]*Z0.shape[1])
    print (dis,tk,rk)
    return (b_Z,h_err)

a = NP.array([[1,0,1,1],[1,1,1,1],[0,1,0,1],[0,0,0,0]])
b = NP.array([[0,1,1],[1,1,1],[1,0,1],[0,0,1]])
(b_Z,h_err) = best_fit(a,b)
print a
print b
print b_Z
print h_err