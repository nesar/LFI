import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import numpy as np
from scipy.stats import skewnorm
import math
import astroabc
from distance_calc import DistanceCalc
from bin_data import *


######################################################################################################

###################### PARAMETERS ##############################
import params

original_dim = params.original_dim  # 2549
latent_dim = params.latent_dim  # 10

ClID = params.ClID
num_train = params.num_train  # 512
num_test = params.num_test  # 32
num_para = params.num_para  # 5

batch_size = params.batch_size  # 8
num_epochs = params.num_epochs  # 100
epsilon_mean = params.epsilon_mean  # 1.0
epsilon_std = params.epsilon_std  # 1.0
learning_rate = params.learning_rate  # 1e-3
decay_rate = params.decay_rate  # 0.0

noise_factor = params.noise_factor  # 0.00

######################## I/O ##################################

DataDir = params.DataDir
PlotsDir = params.PlotsDir
ModelDir = params.ModelDir

fileOut = params.fileOut


#####################################################################################################

#####################################################################################################

def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)

# ----------------------------------------------------------------------------

normFactor = np.loadtxt(DataDir + 'normfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')
meanFactor = np.loadtxt(DataDir + 'meanfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')

print('-------normalization factor:', normFactor)
print('-------rescaling factor:', meanFactor)


#####################################################################################################



# ------------------------------------------------------------------------------
### Using pre-trained GPy model #######################

import GPy
from keras.models import load_model



GPmodelOutfile = DataDir + 'GPy_model' + str(latent_dim) + ClID + fileOut
m1 = GPy.models.GPRegression.load_model(GPmodelOutfile + '.zip')


# decoderFile = ModelDir + 'DecoderP' + str(num_para) + ClID + '_' + fileOut + '.hdf5'
# decoder = load_model(decoderFile)





LoadModel = True
if LoadModel:
    encoder = load_model(ModelDir + 'EncoderP' + str(num_para) + ClID + '_' + fileOut + '.hdf5')
    decoder = load_model(ModelDir + 'DecoderP' + str(num_para) + ClID + '_' + fileOut + '.hdf5')
    history = np.loadtxt(
        ModelDir + 'TrainingHistoryP' + str(num_para) + ClID + '_' + fileOut + '.txt')








########## REAL DATA with ERRORS #############################
# Planck/SPT/WMAP data
# TE, EE, BB next

fileID = 1


dirIn = '../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt', 'PlancklegacyLow.txt']

lID = np.array([0, 2, 0, 0])
ClID = np.array([1, 3, 1, 1])
emaxID = np.array([2, 4, 2, 2])
eminID = np.array([2, 4, 2, 3])

print(allfiles)


# for fileID in [realDataID]:
with open(dirIn + allfiles[fileID]) as f:
    lines = (line for line in f if not line.startswith('#'))
    allCl = np.loadtxt(lines, skiprows=1)

    l = allCl[:, lID[fileID]].astype(int)
    Cl = allCl[:, ClID[fileID]]
    emax = allCl[:, emaxID[fileID]]
    emin = allCl[:, eminID[fileID]]

    print(l.shape)


ls = np.loadtxt(DataDir + 'P' + str(num_para) + 'ls_' + str(num_train) + '.txt')[2:]

x = l[l < ls.max()]
y = Cl[l < ls.max()]
yerr = emax[l < ls.max()]
data = y

######3

# PLANCK low l are not Gaussian errors
#
# So Try there. However, data right now is l>30

###

############## GP FITTING ################################################################################
##########################################################################################################



################# ARCHITECTURE ###############################

# kernel = Matern32Kernel([1000, 4000, 3000, 1000, 2000], ndim=num_para)

# # # ------------------------------------------------------------------------------
# encoded_xtrain = np.loadtxt(
#     DataDir + 'encoded_xtrainP' + str(num_para) + ClID + '_' + fileOut + '.txt').T
# encoded_xtest_original = np.loadtxt(
#     DataDir + 'encoded_xtestP' + str(num_para) + ClID + '_' + fileOut + '.txt')
#


# GPmodelOutfile = DataDir + 'GPy_model' + str(latent_dim) + ClID + fileOut
# m1 = GPy.models.GPRegression.load_model(GPmodelOutfile + '.zip')


def GPyfit(GPmodelOutfile, para_array):


    test_pts = para_array.reshape(num_para, -1).T

    # -------------- Predict latent space ----------------------------------------
    m1p = m1.predict(test_pts)  # [0] is the mean and [1] the predictive
    W_pred = m1p[0]
    # -------------- Decode from latent space --------------------------------------
    x_decoded = decoder.predict(W_pred.reshape(latent_dim, -1).T )
    return (normFactor * x_decoded[0]) + meanFactor



## Make sure the changes are made in log prior definition too. Variable: new_params


param1 = ["$\Omega_c h^2$", 0.1197, 0.10, 0.14] #
# param2 = ["$\Omega_b h^2$", 0.02222, 0.0205, 0.0235]
param3 = ["$\sigma_8$", 0.829, 0.7, 0.9]
# param4 = ["$h$", 0.6731, 0.55, 0.85]
# param5 = ["$n_s$", 0.9655, 0.85, 1.05]



#####################################################################################################


trial_params = np.array([0.119, 0.022, 0.829, 0.6731, 0.965])


x_decodedGPy = GPyfit(GPmodelOutfile, trial_params)


plt.figure(1423)

# plt.plot(x_decoded, 'k--', alpha = 0.4, label = 'George')
plt.plot(x_decodedGPy, '--', alpha = 0.4 , label = 'GPy')
plt.plot(l, Cl, 'r', alpha = 0.3 , label = 'camb')
plt.legend()
plt.show()





#####################################################################################################

#####################################################################################################

#####################################################################################################



#
# def ABCsimulation(param): #param = [om, w0]
#     if param[0] < 0.0 or param[0] > 1.0:
#         return [None]*len(zbins)
#     else:
#         model_1_class = DistanceCalc(param[0],0,1-param[0],0,[param[1],0],0.7)  #om,ok,ol,wmodel,de_params,h0
#         data_abc = np.zeros(len(zbins))
#         for i in range(len(zbins)):
#                 data_abc[i] = model_1_class.mu(zbins[i]) + skewnorm.rvs(a, loc=e, scale=w, size=1)
#         return data_abc
#
#
#
#




def ABCsimulation(param): #param = [om, w0]

    p1, p3 = param
    print(param)

    if p1 < param1[2] or p1 > param1[3] or p3 < param3[2] or p3 > param3[3]:
        return [None] * len(x)

    else:

        new_params = np.array([p1, 0.222, p3, 0.6731, 0.9])
        model = GPyfit(GPmodelOutfile, new_params)  # Using GPy -- using trained model
        mask = np.in1d(ls, x)
        model_mask = model[mask]

        return  model_mask

# #
# def lnprior(theta):
#     p1, p2, p3, p4, p5 = theta
#     # if 0.12 < p1 < 0.155 and 0.7 < p2 < 0.9:
#     if param1[2] < p1 < param1[3] and param2[2] < p2 < param2[3] and param3[2] < p3 < param3[3] \
#             and param4[2] < p4 < param4[3] and param5[2] < p5 < param5[3]:
#         return 0.0
#     return -np.inf



plt.figure(1424)

# plt.plot(x_decoded, 'k--', alpha = 0.4, label = 'George')
# plt.plot(x_decodedGPy, '--', alpha = 0.4 , label = 'GPy')
plt.plot(ABCsimulation([0.119, 0.8]), 'r', label = 'abc_emulator')
plt.plot(Cl, 'ko', label = 'PLANCK')
plt.legend()
plt.show()



#
# nparam = 5
# npart = 100 #number of particles/walkers
# niter = 20  #number of iterations
# tlevels = [500.0,0.005] #maximum,minimum tolerance

#
#
# def my_dist(d,y):
#     if y[0]==None:
#         return float('Inf')
#     else:
#         return np.sum(((y-d)/yerr)**2)
#


def my_dist(d,y):
    if y[0]==None:
        return float('Inf')
    else:
        return np.sum(((y-d)/yerr)**2)




print('my_dist typical value', my_dist(data, ABCsimulation([0.119, 0.8])))


nparam = 2
npart = 10 #100 #number of particles/walkers
niter = 20  #number of iterations
tlevels = [200000.0, 200] #maximum,minimum tolerance




prop={'tol_type':'exp',"verbose":1,'adapt_t':True,
      'threshold':75,'pert_kernel':2,'variance_method':0,
      'dist_type': 'user','dfunc':my_dist, 'restart':"restart_test.txt", \
      'outfile':"abc_pmc_output_"+str(nparam)+"param.txt",'mpi':False,
      'mp':True,'num_proc':2, 'from_restart':False}


# For $\Omega_{m}$ we use a normal distribution with mean $0.3$ and standard deviation $0.5$.
# For $w_{0}$ we use a normal distribution with mean $-1.0$ and standard deviation $0.5$.

# priorname  = ["normal","normal"]
# hyperp = [[0.3,0.5], [-1.0,0.5]]
# prior = zip(priorname,hyperp)




priorname  = ["normal","normal"]#, "normal", "normal", "normal"]
hyperp = [[param1[1], (param1[3] - param1[2])/4.], [param3[1], (param3[3] - param3[2])/4.]]#, [param3[1], (param3[3] - param3[2])/4.], [param4[1], (param4[3] - param4[2])/4.], [param5[1], (param5[3] - param5[2])/4.]]

prior = zip(priorname,hyperp)




sampler = astroabc.ABC_class(nparam,npart,data,tlevels,niter,prior,**prop)
sampler.sample(ABCsimulation)



######### PLOTTING ROUTINE ################

###########################################################################
samples_plot = sampler.chain[:, :, :].reshape((-1, nparam))

print(sampler.outfile)
####### FINAL PARAMETER ESTIMATES #######################################

outfile = sampler.outfile
samples_plot  = np.loadtxt(outfile, skiprows=1)
samples_plot = samples_plot[:, :-2]


import pygtc
fig = pygtc.plotGTC(samples_plot, paramNames=[r'$\Omega_m$', r'$w_1$'], truths= [0.3, -1] ,
                    figureSize='MNRAS_page')