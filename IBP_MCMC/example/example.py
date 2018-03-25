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
from PyIBP import PyIBP as IBP
######################## best-fit###################
def best_fit(Z0,rZ):
    dis,tk = Z0.shape
    rk = rZ.shape[1]
    b_Z = NP.zeros((dis,tk))
    for i in range(tk):
        best_index = 0
        best_dist = dis
        for j in range(rk):
            if  NP.count_nonzero(Z0[:,i]!=rZ[:,j]) < best_dist:
                best_index = j
                best_dist = NP.count_nonzero(Z0[:,i]!=rZ[:,j])

        b_Z [:,i] = rZ [:,best_index]

    h_err = NP.count_nonzero(Z0!=b_Z)/float(Z0.shape[0]*Z0.shape[1])
    return (b_Z,h_err)
######################## Initialize#################
alpha_true = 0.5
numsamp = 24
restart = 25
filename = 'Yeast'
#######################  Make Environment###########
# todir1=filename+'/Hamming-vs-K'
# todir2=filename+'/RMSE-vs-K'
todir3='Real/'+filename+'/RMSEnoise-vs-K'
# if os.path.exists(todir1):
#     shutil.rmtree(todir1)
# os.makedirs(todir1)
# if os.path.exists(todir2):
#     shutil.rmtree(todir2)
# os.makedirs(todir2)
if os.path.exists(todir3):
    shutil.rmtree(todir3)
os.makedirs(todir3)
extra1 = SPIO.loadmat('../../Experiment/Real/'+filename+'/R.mat')
# extra2 = SPIO.loadmat('../../Experiment/'+filename+'/W0.mat')
# extra3 = SPIO.loadmat('../../Experiment/'+filename+'/Z0.mat')
data = extra1['R'] 
# W0 = extra2['W0'] 
# Z0 = extra3['Z0'] 
print 'Handling %s \n' % filename
# file1=open(todir1+'/MCMC','w',0)
# file2=open(todir2+'/MCMC','w',0)
file3=open(todir3+'/MCMC','w',0)
# file1.write('This is %s of hamming\n' % filename)
# file2.write('This is %s of RMSE\n' % filename)
file3.write('This is %s of RMSEnoise\n' % filename)
######################   Start #####################  
# a = 58
# b = 49 
# Load the data
matvals = SPIO.loadmat('block_image_set.mat')
trueWeights = matvals['trueWeights']
features = matvals['features']
# Center the data
(N,D) = data.shape
# cdata = IBP.centerData(data)
for c_alpha in range(5):
    (alpha, alpha_a, alpha_b) = (alpha_true, alpha_true, alpha_true)
    # Observed data Gaussian noise (Gamma hyperparameters)
    (sigma_x, sx_a, sx_b) = (1., 1., 1.)
    # Latent feature weight Gaussian noise (Gamma hyperparameters)
    (sigma_a, sa_a, sa_b) = (1., 1., 1.)    
    start = time.time()
    # Do inference
    for re in range(restart):
        f = IBP(data,(alpha,alpha_a,alpha_b),
            (sigma_x, sx_a, sx_b),
            (sigma_a, sa_a, sa_b))
        broken_flag = 0
        pZ = f.ZV 
        pW = f.weights() 
        for s in range(numsamp):
            try:
                f.sampleReport(s)
            except NP.linalg.linalg.LinAlgError as err:
                print("broken,new restart")
                broken_flag = 1
                break                              
                raise 
            pZ = f.ZV 
            pW = f.weights() 
            f.fullSample()           
            print (filename+" At restart %d iter %d error=%lf ,alpha=%lf " %(re,s,pow(NP.linalg.norm(data-NP.dot(f.ZV,f.weights())),2),alpha_true))
            if s == 0 :
                best_iter = 0
                best_iter_k = f.ZV.shape[1]
                best_iter_error = pow(NP.linalg.norm(data-NP.dot(f.ZV,f.weights())),2)
                best_iter_z = f.ZV
                best_iter_w = f.weights()
            elif  pow(NP.linalg.norm(data-NP.dot(f.ZV,f.weights())),2) < best_iter_error :
                best_iter = s
                best_iter_k = f.ZV.shape[1]
                best_iter_error = pow(NP.linalg.norm(data-NP.dot(f.ZV,f.weights())),2)
                best_iter_z = f.ZV
                best_iter_w = f.weights()
            print ("At restart %d best at iter %d k=%d error=%lf " %(re,best_iter,best_iter_k,best_iter_error))
            if re > 0 :
                print ("Optimal at restart %d alpha=%lf k=%d error=%lf " %(best_re,alpha_true,best_k,best_error))
        # May try catch to here ,try to recover the previous iteration
        if broken_flag == 1 :
            best_iter = s-1
            best_iter_k = pZ.shape[1]
            best_iter_error = pow(NP.linalg.norm(data-NP.dot(pZ,pW)),2)
            best_iter_z = pZ
            best_iter_w = pW
        # Write result to file /reconstruct best fit Z
        if best_iter_k != 0 :
            # (b_Z,h_err) = best_fit(Z0,best_iter_z)
            # rmse_err=NP.linalg.norm(NP.dot(Z0,W0)-NP.dot(best_iter_z,best_iter_w))/float(math.sqrt(N*D))
            rmse_n_err=math.sqrt(best_iter_error)/float(math.sqrt(N*D))
            # file1.write("%d %g\n" %(best_iter_k,h_err))
            # file2.write("%d %g\n" %(best_iter_k,rmse_err))
            file3.write("%d %g\n" %(best_iter_k,rmse_n_err))        
        try:
            import matplotlib.pyplot as P
            from scaledimage import scaledimage
        except:
            print ('matplotlib not installed, skipping visualization...')
            sys.exit(0)   
        print (best_iter_z.shape)
        print (best_iter_w.shape)
        print (data.shape )
        # Intensity plots of
        # -ground truth factor-feature weights (top)
        # -learned factor-feature weights (bottom)
        # K = max(len(trueWeights), len(best_iter_w))
        # (fig, subaxes) = P.subplots(2, K)
        # for sa in subaxes.flatten():
        #     sa.set_visible(False)
        # combination = 'runs with alpha=%lf,restart at %d,k=%d,error=%e' %(alpha_true,re,best_iter_k,best_iter_error)
        # fig.suptitle(combination)   
        # for (idx, learnedFactor) in enumerate(best_iter_w):
        #     ax = subaxes[0, idx]    
        #     scaledimage(learnedFactor.reshape(a,b),
        #                 pixwidth=3, ax=ax)
        #     ax.set_visible(True)
        # ax = subaxes[1, 0]    
        # scaledimage(data.mean(axis=0).reshape(a,b),pixwidth=3, ax=ax)
        # ax.set_visible(True)
        # filename = filename + '_alpha_%lf_re_%d' %(alpha_true,re)
        # P.savefig(filename+'.png')
        #try to maintain best fit among restarts         
        if re == 0 :
            best_re = 0
            best_k = best_iter_z.shape[1]
            best_error = pow(NP.linalg.norm(data-NP.dot(best_iter_z,best_iter_w)),2)
            best_z = best_iter_z
            best_w = best_iter_w
        elif  pow(NP.linalg.norm(data-NP.dot(best_iter_z,best_iter_w)),2) < best_error :
            best_re = re
            best_k = best_iter_z.shape[1]
            best_error = pow(NP.linalg.norm(data-NP.dot(best_iter_z,best_iter_w)),2)
            best_z = best_iter_z
            best_w = best_iter_w

    end = time.time()
    print ("The best reconstruction error occurs at re-start %d with k= %d error = %lf , training time = %lf" %(best_re,best_k,best_error,(end-start)*(best_re+1)/(restart)))
    alpha_true=alpha_true*2.0
end