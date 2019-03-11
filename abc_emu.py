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


zbins,avmu_bin,averr_bin,mu_in_bin_new,mu_in_bin_new = read_data()


e = -0.1 #location
w = 0.3 #scale
a = 5.0 #skew

plt.figure(figsize=(17,8))
plt.hist(skewnorm.rvs(a, loc=e, scale=w, size=10000),normed=True,bins=20,color='#593686')
plt.title("Distribution of a random sample",fontsize=17);



data = np.zeros(len(zbins))

for i in range(len(zbins)):
    data[i] = avmu_bin[i] + skewnorm.rvs(a, loc=e, scale=w, size=1)


plt.figure(figsize=(17,8))
plt.errorbar(zbins,avmu_bin,averr_bin,marker="o",linestyle="None",label="without noise",color='#593686')
plt.scatter(zbins,data,color='r',label="with noise")
plt.legend(loc="upper left",prop={'size':17});
plt.xlabel("$z$",fontsize=20)
plt.ylabel("$\mu(z)$",fontsize=20)
plt.title("Data before and after noise is added",fontsize=17);


z = 0
distribution = np.zeros(10000)

for j in range(10000):
    distribution[j] = avmu_bin[z] + skewnorm.rvs(a, loc=e, scale=w, size=1)

plt.figure(figsize=(17,8))
plt.title("Distribution of the data at redshift z=0.5",fontsize=17);
plt.hist(distribution,bins=20,color='#593686',normed=True)
plt.plot((avmu_bin[z], avmu_bin[z]), (0, 2.5), 'r-', label="True $\mu$ at $z = 0.5$");
plt.legend(prop={'size':16});



def my_dist(d,x):
    if x[0]==None:
        return float('Inf')
    else:
        return np.sum(((x-d)/averr_bin)**2)



nparam = 2
npart = 100 #number of particles/walkers
niter = 20  #number of iterations
tlevels = [500.0,0.005] #maximum,minimum tolerance

prop={'tol_type':'exp',"verbose":1,'adapt_t':True,
      'threshold':75,'pert_kernel':2,'variance_method':0,
      'dist_type': 'user','dfunc':my_dist, 'restart':"restart_test.txt", \
      'outfile':"abc_pmc_output_"+str(nparam)+"param.txt",'mpi':False,
      'mp':True,'num_proc':2, 'from_restart':False}


priorname  = ["normal","normal"]
hyperp = [[0.3,0.5], [-1.0,0.5]]
prior = zip(priorname,hyperp)



def ABCsimulation(param): #param = [om, w0]
    if param[0] < 0.0 or param[0] > 1.0:
        return [None]*len(zbins)
    else:
        model_1_class = DistanceCalc(param[0],0,1-param[0],0,[param[1],0],0.7)  #om,ok,ol,wmodel,de_params,h0
        data_abc = np.zeros(len(zbins))
        for i in range(len(zbins)):
                data_abc[i] = model_1_class.mu(zbins[i]) + skewnorm.rvs(a, loc=e, scale=w, size=1)
        return data_abc








def ABCsimulation(param): #param = [om, w0]
    if param[0] < 0.0 or param[0] > 1.0:
        return [None]*len(zbins)
    else:
        model_1_class = DistanceCalc(param[0],0,1-param[0],0,[param[1],0],0.7)  #om,ok,ol,wmodel,de_params,h0
        data_abc = np.zeros(len(zbins))
        for i in range(len(zbins)):
                data_abc[i] = model_1_class.mu(zbins[i]) + skewnorm.rvs(a, loc=e, scale=w, size=1)
        return data_abc
















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