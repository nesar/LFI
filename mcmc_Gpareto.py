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

ndim = 2
nwalkers = 100  # 200 #600  # 500
nrun_burn = 10  # 50 # 50  # 300
nrun = 50  # 300  # 700
fileID = 1

###### for ell < 30
ndim = 2
nwalkers= 100 #100 #number of particles/walkers
nrun = 75  #number of iterations


########## REAL DATA with ERRORS #############################
# Planck/SPT/WMAP data
# TE, EE, BB next


fileID = 3


dirIn = '../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt', 'PlancklegacyLow_ell29.txt',
            'PlancklegacyLow_all.txt']

lID = np.array([0, 2, 0, 0, 0])
ClID = np.array([1, 3, 1, 1, 1])
emaxID = np.array([2, 4, 2, 2, 2])
eminID = np.array([2, 4, 2, 3, 3])

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


############## GP FITTING ################################################################################
##########################################################################################################



def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


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
print np.abs(Pxx - (np.dot(U, np.dot(np.diag(s), Vh))) ).max()



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


########################################################################################################################
########################################################################################################################

#### Cosmological Parameters ########################################


param1 = ["$\Omega_c h^2$", 0.1197, 0.10, 0.14] #
# param2 = ["$\Omega_b h^2$", 0.02222, 0.0205, 0.0235]
param3 = ["$\sigma_8$", 0.829, 0.7, 0.9]
# param4 = ["$h$", 0.6731, 0.55, 0.85]
# param5 = ["$n_s$", 0.9655, 0.85, 1.05]



#################### CHAIN INITIALIZATION ##########################

## 2 options

Uniform_init = True
if Uniform_init:
    # Choice 1: chain uniformly distributed in the range of the parameters
    pos_min = np.array([param1[2], param3[2]])
    pos_max = np.array([param1[3], param3[3]])
    psize = pos_max - pos_min
    pos0 = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]

True_init = False
if True_init:
    # Choice 2: chain is initialized in a tight ball around the expected values
    pos0 = [[param1[1] * 1.2, param3[1] * 0.9] +
            1e-3 * np.random.randn(ndim) for i in range(nwalkers)]

MaxLikelihood_init = False
if MaxLikelihood_init:
    # Choice 2b: Find expected values from max likelihood and use that for chain initialization
    # Requires likehood function below to run first

    import scipy.optimize as op

    nll = lambda *args: -lnlike(*args)
    result = op.minimize(nll, [param1[1], param2[1], param3[1], param4[1], param5[1]],
                         args=(x, y, yerr))
    p1_ml, p2_ml, p3_ml, p4_ml, p5_ml = result["x"]
    print result['x']

    pos0 = [result['x'] + 1.e-4 * np.random.randn(ndim) for i in range(nwalkers)]

# Visualize the initialization

PriorPlot = False

if PriorPlot:
    fig = corner.corner(pos0, labels=[param1[0], param2[0], param3[0], param4[0], param5[0]],
                        range=[[param1[2], param1[3]], [param2[2], param2[3]],
                               [param3[2], param3[3]],
                               [param4[2], param4[3]], [param5[2], param5[3]]],
                        truths=[param1[1], param2[1], param3[1], param4[1], param5[1]])
    fig.set_size_inches(10, 10)

######### MCMC #######################


x = l[l < ls.max()]
y = Cl[l < ls.max()]
yerr = emax[l < ls.max()]


## Sample implementation :
# http://eso-python.github.io/ESOPythonTutorials/ESOPythonDemoDay8_MCMC_with_emcee.html
# https://users.obs.carnegiescience.edu/cburns/ipynbs/Emcee.html

def lnprior(theta):
    # p1, p2, p3, p4, p5 = theta
    p1, p3 = theta

    # if 0.12 < p1 < 0.155 and 0.7 < p2 < 0.9:
    if param1[2] < p1 < param1[3] and param3[2] < p3 < param3[3]:
            # and param4[2] < p4 < param4[3] and param5[2] < p5 < param5[3]:
        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr):
    p1, p3 = theta
    # new_params = np.array([p1, 0.0225, p2 , 0.74, 0.9])

    new_params = np.array([p1, 0.022, p3, 0.6731, 0.9])
    # model = GPfit(computedGP, new_params)#  Using George -- with model training

    model = GP_fit(new_params)  # Using GPy -- using trained model

    mask = np.in1d(ls, x)
    model_mask = model[mask]

    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model_mask) / yerr) ** 2.))


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


# Let us setup the emcee Ensemble Sampler
# It is very simple: just one, self-explanatory line

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

###### BURIN-IN #################

time0 = time.time()
# burnin phase
pos, prob, state = sampler.run_mcmc(pos0, nrun_burn)
sampler.reset()
time1 = time.time()
print('burn-in time:', time1 - time0)

###### MCMC ##################
time0 = time.time()
# perform MCMC
pos, prob, state = sampler.run_mcmc(pos, nrun)
time1 = time.time()
print('mcmc time:', time1 - time0)

samples = sampler.flatchain
samples.shape

###########################################################################
samples_plot = sampler.chain[:, :, :].reshape((-1, ndim))

np.savetxt('mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
    nrun) + ClID + '_' + fileOut + allfiles[fileID][:-4] + '.txt',
           sampler.chain[:, :, :].reshape((-1, ndim)))

####### FINAL PARAMETER ESTIMATES #######################################


samples_plot = np.loadtxt('mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) +
                          '_run' + str(nrun) + ClID + '_' + fileOut + allfiles[fileID][
                                                                      :-4] + '.txt')

# samples = np.exp(samples)
p1_mcmc, p3_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))
print('mcmc results:', p1_mcmc[0],p3_mcmc[0])

####### CORNER PLOT ESTIMATES #######################################

fig = pygtc.plotGTC(samples_plot, paramNames=[param1[0], param3[0]], truths=[0.1187,  0.802], figureSize='MNRAS_page', nContourLevels=3)  # , plotDensity
# = True, filledPlots = False,\smoothingKernel = 0, nContourLevels=3)

fig.savefig('mcmc_pygtc_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(nrun) + ClID +
            '_' + fileOut + allfiles[fileID][:-4] + '.pdf')

####### FINAL PARAMETER ESTIMATES #######################################
#
# plt.figure(1432)
#
# xl = np.array([0, 10])
# for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
#     plt.plot(xl, m*xl+b, color="k", alpha=0.1)
# plt.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
# plt.errorbar(x, y, yerr=yerr, fmt=".k", alpha=0.1)



####### SAMPLER CONVERGENCE #######################################

ConvergePlot = False
if ConvergePlot:
    fig = plt.figure(13214)
    plt.xlabel('steps')
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    ax5 = fig.add_subplot(5, 1, 5)

    ax1.plot(np.arange(nrun), sampler.chain[:, :, 0].T, lw=0.2, alpha=0.9)
    ax1.text(0.9, 0.9, param1[0], horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes, fontsize=20)
    ax2.plot(np.arange(nrun), sampler.chain[:, :, 1].T, lw=0.2, alpha=0.9)
    ax2.text(0.9, 0.9, param2[0], horizontalalignment='center', verticalalignment='center',
             transform=ax2.transAxes, fontsize=20)
    ax3.plot(np.arange(nrun), sampler.chain[:, :, 2].T, lw=0.2, alpha=0.9)
    ax3.text(0.9, 0.9, param3[0], horizontalalignment='center', verticalalignment='center',
             transform=ax3.transAxes, fontsize=20)
    ax4.plot(np.arange(nrun), sampler.chain[:, :, 3].T, lw=0.2, alpha=0.9)
    ax4.text(0.9, 0.9, param4[0], horizontalalignment='center', verticalalignment='center',
             transform=ax4.transAxes, fontsize=20)
    ax5.plot(np.arange(nrun), sampler.chain[:, :, 4].T, lw=0.2, alpha=0.9)
    ax5.text(0.9, 0.9, param5[0], horizontalalignment='center', verticalalignment='center',
             transform=ax5.transAxes, fontsize=20)
    plt.show()

    fig.savefig(PlotsDir + 'convergencePCA_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
        nrun) + ClID + '_' + fileOut + allfiles[fileID][:-4] + '.pdf')