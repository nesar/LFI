import numpy as np

# import matplotlib as mpl
# mpl.use('Agg')


import matplotlib.pylab as plt
import corner
import emcee
import time
# from keras.models import load_model
import params
# import george
# from george.kernels import Matern32Kernel

import rpy2.robjects as ro

import pygtc
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

#### parameters that define the MCMC

# ndim = 5
# nwalkers = 20  # 200 #600  # 500
# nrun_burn = 10  # 50 # 50  # 300
nrun = 30  # 300  # 700



###################### PARAMETERS ##############################

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



# ----------------------------- i/o ------------------------------------------

# ClID = str(ClID[fileID])   ### UGH dirty fix
#
# Trainfiles = np.loadtxt(DataDir + 'P' + str(num_para) + ClID + 'Cl_' + str(num_train) + '.txt')
Testfiles = np.loadtxt(DataDir + 'P' + str(num_para) + ClID + 'Cl_' + str(num_test) + '.txt')
#
# x_train = Trainfiles[:, num_para + 2:]
x_test = Testfiles[:, num_para + 2:]
# y_train = Trainfiles[:, 0: num_para]
y_test = Testfiles[:, 0: num_para]
#

#
ls = np.loadtxt(DataDir + 'P' + str(num_para) + 'ls_' + str(num_train) + '.txt')[2:]
#
# # ----------------------------------------------------------------------------
#
normFactor = np.loadtxt(DataDir + 'normfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')
meanFactor = np.loadtxt(DataDir + 'meanfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')

########## REAL DATA with ERRORS #############################
# Planck/SPT/WMAP data
# TE, EE, BB next

fileID = 3


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



######### NEED TO CHANGE THIS LOW Planck values ###########
# if fileID == 3:
#     # PlancklegacyLow.txt
#     ls = np.arange(ls.min(), 30, 1)
#


x = l[l < ls.max()]
y = Cl[l < ls.max()]
yerr = emax[l < ls.max()]
data = y



############## GP FITTING ################################################################################
##########################################################################################################



def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)




import numpy as np
from rpy2.robjects import r
from rpy2.robjects.packages import importr

RcppCNPy = importr('RcppCNPy')
# RcppCNPy.chooseCRANmirror(ind=1) # select the first mirror in the list

########

################################# I/O #################################

# Note that the 3rd variable is not used here, and the first two points of the spectrum can be removed
r(
    'u_train2 <- as.matrix(read.csv("../Cl_data/Data/LatinCosmoP51024.txt", sep = " ", header = ''F))')

r(
    'y_train2 <- as.matrix(read.csv("../Cl_data/Data/P5TTCl_1024.txt", sep = " ", header = ''F))[,''-(1:7)]')

# r(
#     'u_test2 <- as.matrix(read.csv("ComparisonTests/VAE_data/params.txt", sep = " ", header = F))')  ## testing design
#
# r(
#     'y_test2 <- as.matrix(read.csv("ComparisonTests/VAE_data/TTtrue.txt", sep = " ", header = F))')  # [,-c(1,2)] # testing spectrum curves

# r('matplot(t(y_train2), type = "l")')

u_train2 = np.loadtxt("../Cl_data/Data/LatinCosmoP51024.txt")
y_train2 = np.loadtxt("../Cl_data/Data/P5TTCl_1024.txt")[:, 7:]
# u_test2 = np.loadtxt("ComparisonTests/VAE_data/params.txt")
# y_test2 = np.loadtxt("ComparisonTests/VAE_data/TTtrue.txt")



########################### PCA ###################################


Dicekriging = importr('DiceKriging')

r('require(foreach)')

r('svd(y_train2)')

r('nrankmax <- 4')  ## Number of components

r('svd_decomp2 <- svd(y_train2)')
r('svd_weights2 <- svd_decomp2$u[, 1:nrankmax] %*% diag(svd_decomp2$d[1:nrankmax])')



import scipy.linalg as SL
nrankmax = 4

y = y_train2.T
yRowMean = np.zeros_like(y[:,0])

for i in range(y.shape[0]):
    yRowMean[i] = np.mean(y[i])

for i in range( y[0].shape[0] ):
    y[:,i] = (y[:,i] - yRowMean)

stdy = np.std(y)
y = y/stdy

Pxx = y
U, s, Vh = SL.svd(Pxx, full_matrices=False)
# assert np.allclose(Pxx, np.dot(U, np.dot(np.diag(s), Vh)))
print(np.abs(Pxx - (np.dot(U, np.dot(np.diag(s), Vh))) ).max())



TruncU = U[:, :nrankmax]     #Truncation
TruncS = s[:nrankmax]
TruncSq = np.diag(TruncS)
TruncVh = Vh[:nrankmax,:]

K = np.matmul(TruncU, TruncSq)/np.sqrt(nrankmax)
W1 = np.sqrt(nrankmax)*np.matmul(np.diag(1./TruncS), TruncU.T)
W = np.matmul(W1, y)

Pred = np.matmul(K,W)


plt.figure(123)
plt.plot(Pred.T[0], 'k')
plt.plot(Pxx.T[0], '--')

plt.show()
########################### GP train #################################

## Build GP models
GPareto = importr('GPareto')

r('''if(file.exists("abcR_GP_models.RData")){
        load("abcR_GP_models.RData")
    }else{
        models_svd2 <- list()
        for (i in 1: nrankmax){
            mod_s <- km(~., design = u_train2, response = svd_weights2[, i])
            models_svd2 <- c(models_svd2, list(mod_s))
        }
        save(models_svd2, file = "abcR_GP_models.RData")

     }''')

r('''''')

#########################################################################################
## All GP fitting together -- workes fine, except we get one value of variance for all
# output dimensions, since they're considered independant

#
# import GPy
#
# kern = GPy.kern.Matern52(input_dim=num_para)
#
# m1 = GPy.models.GPRegression(W, u_train2, kernel=kern)
# m1.Gaussian_noise.variance.constrain_fixed(1e-12)
# m1.optimize(messages=True)
# m1p = m1.predict(x_test)  # [0] is the mean and [1] the predictive
# W_predArray = m1p[0]
#
# np.savetxt(DataDir + 'WPredArray_GPyNoVariance' + str(latent_dim) + '.txt', W_predArray)


######################### INFERENCE ########################

# exit()

def GP_fit(para_array):
    # test_pts = para_array.reshape(num_para, -1).T
    # print(test_pts.shape)

    test_pts = para_array
    test_pts = np.expand_dims(test_pts, axis=0)

    # # -------------- Predict latent space ----------------------------------------
    #
    # # W_pred = np.array([np.zeros(shape=latent_dim)])
    # # W_pred_var = np.array([np.zeros(shape=latent_dim)])
    #
    # m1p = m1.predict(test_pts)  # [0] is the mean and [1] the predictive
    # W_pred = m1p[0]
    # # W_varArray = m1p[1]
    #
    #
    # # for j in range(latent_dim):
    # #     W_pred[:, j], W_pred_var[:, j] = computedGP["fit{0}".format(j)].predict(encoded_xtrain[j],
    # #                                                                             test_pts)
    #
    # # -------------- Decode from latent space --------------------------------------
    #
    # x_decoded = decoder.predict(W_pred.reshape(latent_dim, -1).T )
    #
    # return (normFactor * x_decoded[0]) + meanFactor

    B = test_pts

    nr, nc = B.shape
    Br = ro.r.matrix(B, nrow=nr, ncol=nc)

    ro.r.assign("B", Br)

    r('wtestsvd2 <- predict_kms(models_svd2, newdata = B , type = "UK")')
    r('reconst_s2 <- t(wtestsvd2$mean) %*% t(svd_decomp2$v[,1:nrankmax])')

    y_recon = np.array(r('reconst_s2'))

    return y_recon[0]



x_id = 20

x_decodedGPy = GP_fit(y_test[x_id])
# computedGP = GPcompute(rescaledTrainParams, latent_dim)
# x_decoded = GPfit(computedGP, y_test[x_id])

# x_camb = (normFactor * x_test[x_id]) + meanFactor


plt.figure(1423)

# plt.plot(x_decoded, 'k--', alpha = 0.4, label = 'George')
plt.plot(x_decodedGPy, alpha = 0.4 , ls = '--', label = 'GPy')
plt.plot(x_test[x_id], alpha = 0.3 , label = 'camb')
plt.legend()
plt.show()


# import sys
# sys.exit()




#####################################################################################################




## Make sure the changes are made in log prior definition too. Variable: new_params


param1 = ["$\Omega_c h^2$", 0.1197, 0.10, 0.14] #
# param2 = ["$\Omega_b h^2$", 0.02222, 0.0205, 0.0235]
param3 = ["$\sigma_8$", 0.829, 0.7, 0.9]
# param4 = ["$h$", 0.6731, 0.55, 0.85]
# param5 = ["$n_s$", 0.9655, 0.85, 1.05]



#####################################################################################################


trial_params = np.array([0.119, 0.022, 0.829, 0.6731, 0.965])


x_decodedGPy = GP_fit(trial_params)


plt.figure(1423)

# plt.plot(x_decoded, 'k--', alpha = 0.4, label = 'George')
plt.plot(x_decodedGPy, '--', alpha = 0.4 , label = 'GPy')
plt.plot(l, Cl, 'r', alpha = 0.3 , label = 'camb')
plt.legend()
plt.show()







def ABCsimulation(param): #param = [om, w0]

    p1, p3 = param
    print(param)

    if p1 < param1[2] or p1 > param1[3] or p3 < param3[2] or p3 > param3[3]:
        return [None] * len(x)

    else:

        new_params = np.array([p1, 0.022, p3, 0.6731, 0.9])
        model = GP_fit(new_params)  # Using GPy -- using trained model
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
plt.plot(x_decodedGPy, '--', alpha = 0.4 , label = 'GPy')
# plt.plot(ABCsimulation([0.119, 0.8]), 'rx', alpha = 0.3,  label = 'abc_emulator')
plt.plot(Cl, 'ko', label = 'PLANCK', alpha = 0.1)
plt.legend()
plt.show()


def my_dist(d,y):
    if y[0]==None:
        return float('Inf')
    else:
        return np.sum(((y-d)/yerr)**2)
        # return np.min(((y - d) / yerr) ** 2)


print('my_dist typical value', my_dist(data, ABCsimulation([0.119, 0.8])))

plt.figure(3)
plt.plot(data)
plt.plot(ABCsimulation([0.119, 0.8]))
new_params = np.array([0.119, 0.022, 0.8, 0.6731, 0.9])
plt.plot(GP_fit(new_params))  # Using GPy -- using trained model
plt.show()


nparam = 2
npart = 100 #100 #number of particles/walkers
niter = 50  #number of iterations
# tlevels = [200000.0, 200] #maximum,minimum tolerance
# tlevels = [500.0, 0.005] #maximum,minimum tolerance
# tlevels = [1e-6,1e-12] #maximum,minimum tolerance
# tlevels = [5000.0, 0.05] #maximum,minimum tolerance
tlevels = [5000.0, 0.5] #maximum,minimum tolerance





prop={'tol_type':'exp',"verbose":0,'adapt_t':True,
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


import astroabc


sampler = astroabc.ABC_class(nparam,npart,data,tlevels,niter,prior,**prop)
sampler.sample(ABCsimulation)



######### PLOTTING ROUTINE ################

###########################################################################
# samples_plot = sampler.chain[:, :, :].reshape((-1, nparam))

print(sampler.outfile)
####### FINAL PARAMETER ESTIMATES #######################################

outfile = sampler.outfile
samples_plot  = np.loadtxt(outfile, skiprows=1)
samples_plot = samples_plot[:, :-2]


# outfile = 'restart_test.txt'
# samples_plot  = np.loadtxt(outfile, skiprows=1)


import pygtc
fig = pygtc.plotGTC(samples_plot, paramNames=[r'$\Omega_m$', r'$\sigma_8$'], nContourLevels = 3, truths= [0.1187,  0.802] ,figureSize='MNRAS_page', plotName='abcPlanck.pdf')

fig.savefig('abc_pygtc_' + str(nparam) + '_nwalk' + str(npart) + '_run' + str(niter) + ClID +
            '_' + fileOut + allfiles[fileID][:-4] + '.pdf')



################################################################3

CompareMCMC = True


if CompareMCMC:
    import numpy as np
    import pygtc

    # nparam = 2
    # npart = 100  # 100 #number of particles/walkers
    # niter = 50  # number of iterations
    #
    # ndim = nparam
    # nwalkers = npart
    # nrun = niter
    #
    # samples_mcmc = np.loadtxt('mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) +
    #                           '_run' + str(nrun) + ClID + '_' + fileOut + allfiles[fileID][
    #                                                                       :-4] + '.txt')

    samples_mcmc = np.loadtxt('mcmc_ndim2_nwalk100_run50TT_P5Model_tot1024_batch8_lr0.0001_decay1'
                              '.0_z32_epoch7500PlancklegacyLow.txt')

    samples_abc = np.loadtxt('abc_pmc_output_2param.txt', skiprows=1)[:, :-2]

    # samples = np.exp(samples)
    p1_mcmc, p3_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                           zip(*np.percentile(samples_mcmc, [16, 50, 84], axis=0)))
    print('mcmc results:', p1_mcmc[0], p3_mcmc[0])

    ####### CORNER PLOT ESTIMATES #######################################



    fig = pygtc.plotGTC(chains=[samples_mcmc, samples_abc], paramNames=[r'$\Omega_m$',
                                                                        r'$\sigma_8$'], truths= [
        0.1187,  0.802], colorsOrder=('blues', 'reds'), chainLabels=["MCMC", "ABC-SMC"], nContourLevels=2, filledPlots = False, figureSize='MNRAS_page')  # , plotDensity = True, filledPlots = False,\smoothingKernel = 0,
    # nContourLevels=3)


    #
    # fig = pygtc.plotGTC(samples_mcmc, paramNames=[param1[0], param3[0]], truths=[0.1187, 0.802],
    #                     figureSize='MNRAS_page')  # , plotDensity
    # # = True, filledPlots = False,\smoothingKernel = 0, nContourLevels=3)

    fig.savefig('compare_pygtc.pdf')
