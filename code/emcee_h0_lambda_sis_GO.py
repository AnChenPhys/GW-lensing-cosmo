import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
import scipy.constants as constants
from scipy.stats import gaussian_kde
from lal import MTSUN_SI, PC_SI
from astropy.cosmology import FlatLambdaCDM
import random
import json
import emcee
import corner

def t_delay_geom_plus(y):
    return -y-0.5
def t_delay_geom_minus(y):
    return y-0.5
def DeltaT(y):
    return t_delay_geom_minus(y)-t_delay_geom_plus(y)

def t_delay(H0,Om0,zL,zS,thetaE,y):
    cosmo_H0 = FlatLambdaCDM(H0=H0,Om0=Om0)
    DL = cosmo_H0.angular_diameter_distance(zL).value
    DS = cosmo_H0.angular_diameter_distance(zS).value
    DLS = cosmo_H0.angular_diameter_distance_z1z2(zL,zS).value

    return (1+zL)/constants.c * DL*DS/DLS*1e6*PC_SI*thetaE**2 *DeltaT(y)

with open('sampling_sis_GO_seed_0_ind_5081/label_result.json', 'r') as file: 
    data = json.load(file)

tc_plus = np.array(data['posterior']['content']['tc_plus'])*1e-4
tc_minus = np.array(data['posterior']['content']['tc_minus'])*1e-4
tc_diff = np.mean(tc_minus) - np.mean(tc_plus)
sigma_t_sq = np.std(tc_plus)**2 + np.std(tc_minus)**2

kde_dL = gaussian_kde(data['posterior']['content']['DL'])
dL_arr = np.linspace(min(data['posterior']['content']['DL']),max(data['posterior']['content']['DL']),10000)
p_dL_rmp = kde_dL(dL_arr)/dL_arr**2
dL_data_rw = random.choices(dL_arr,weights=p_dL_rmp,k=10000)
dL_mean = np.mean(dL_data_rw)
dL_std = np.std(dL_data_rw)

ML = 5.34e10
yh = 0.315
zL = 0.607
zS = 1.044
t_delay_obs = DeltaT(yh)*4*np.pi*ML*(1+zL)*MTSUN_SI + tc_diff

H0 = 67.7
Om0 = 0.308
cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)

DL = cosmo.angular_diameter_distance(zL).value
DS = cosmo.angular_diameter_distance(zS).value
DLS = cosmo.angular_diameter_distance_z1z2(zL,zS).value
thetaE = np.sqrt(4*np.pi*ML*MTSUN_SI*constants.c/PC_SI *DLS/DS/DL/1e6)

def Loglik(free_param, dL_mean, dL_std, t_delay_obs, sigma_t_sq,zL,zS,thetaE,yh):
    loglik = 0
    H0 = free_param[0]
    # Om0 = free_param[1]
    # zS = free_param[1]
    lamb = free_param[1]

    if 10<=H0<=200 and 0.05<=Om0<=0.95 and 0.5<=lamb<=2:
        cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)
        loglik += -0.5*(dL_mean - cosmo.luminosity_distance(zS).value)**2/dL_std**2

        t_delay_thy = t_delay(H0,Om0,zL,zS,thetaE,yh)*lamb
        loglik += -0.5*(t_delay_obs - t_delay_thy)**2/sigma_t_sq**2
    else:
        loglik = -np.inf

    return loglik * 1e-25


# h0_arr = np.linspace(20,140,100)
# lamb_arr = np.linspace(0.5,2,100)
# lik = np.zeros((len(h0_arr),len(lamb_arr)))
# for i in range(len(h0_arr)):
#     for j in range(len(lamb_arr)):
#         lik[i,j] = Loglik([h0_arr[i],lamb_arr[j]], dL_mean, dL_std, t_delay_obs, sigma_t_sq,zL,zS,thetaE,yh)
# np.savetxt('H0_lambda_loglik_seed_0_ind_5081.txt',lik)
# # lik = np.loadtxt('H0_lambda_loglik_seed_0_ind_5081.txt')
# lik *= 1e-25

# ndim = 2
# param_values = [h0_arr,lamb_arr]
# label = [r'$H_0$',r'$\lambda$']
# fig, ax = plt.subplots(ndim, ndim, figsize=[15,15],constrained_layout=True)

# from utils import *
# for column in np.arange(0,ndim):
#     for row in np.arange(0,ndim):
#         indices = list(range(ndim))
#         if column > row:
#             fig.delaxes(ax[row][column])

#         elif row == column:
#             indices.remove(row)
#             ind_lik = np.sum(np.exp(lik),axis=tuple(indices))
#             ind_lik_norm = ind_lik/np.sum(ind_lik)/(param_values[row][1]-param_values[row][0])

#             ax[row,column].plot(param_values[row],ind_lik_norm)
#             ax[row,column].set_xlim(param_values[row][0],param_values[row][-1])
#             ax[row,column].set_ylim(0,1.2*np.max(ind_lik_norm))

#             con_int = confidence_interval(ind_lik_norm, param_values[row])
#             ax[row,column].text(0.5,0.95, '%.2f + %.2f - %.2f'%(con_int.map, con_int.upper_level-con_int.map, con_int.map-con_int.lower_level), fontsize=15, horizontalalignment='center',verticalalignment='center', transform=ax[row,column].transAxes)

#             ax[0,0].set_xlim(20,140)
#             ax[1,1].set_xlim(0.5,2)
#             ax[row,column].xaxis.set_tick_params(labelsize=15)
#             ax[row,column].yaxis.set_tick_params(labelsize=15)

#         else:
#             indices.remove(row)
#             indices.remove(column)
#             ax[row,column].contourf(param_values[column],param_values[row],np.sum(np.exp(lik),axis=tuple(indices)).T,20)
#             ax[row,column].set_xlim(20,140)
#             ax[row,column].set_ylim(0.5,2)
#             ax[row,column].xaxis.set_tick_params(labelsize=15)
#             ax[row,column].yaxis.set_tick_params(labelsize=15)

#         if column == 0:
#             ax[row,column].set_ylabel(label[row], fontsize=16)
#         if row == ndim-1:
#             ax[row,column].set_xlabel(label[column],fontsize=16)

#         ax[0,0].axvline(H0,color='grey',linestyle='--')
#         ax[1,1].axvline(1,color='grey',linestyle='--')

# plt.savefig('../plots/lik_h0_lamb.png')

# lik_h0 = np.zeros(100)
# lik_lamb = np.zeros(100)
# for i in range(100):
#     lik_h0[i] = Loglik([h0_arr[i],1], dL_mean, dL_std, t_delay_obs, sigma_t_sq,zL,zS,thetaE,yh)
#     lik_lamb[i] = Loglik([H0,lamb_arr[i]], dL_mean, dL_std, t_delay_obs, sigma_t_sq,zL,zS,thetaE,yh)
# plt.figure()
# plt.grid()
# plt.plot(h0_arr,lik_h0)
# plt.savefig('../plots/lik_h0.png')

# plt.figure()
# plt.grid()
# plt.plot(lamb_arr,lik_lamb)
# plt.savefig('../plots/lik_lamb.png')

nwalkers = 32
ndim = 2
p0 = np.random.rand(nwalkers, ndim)*np.array([1.2,1.5]) + np.array([0.2,0.5])

sampler = emcee.EnsembleSampler(nwalkers, ndim, Loglik, args=[dL_mean, dL_std, t_delay_obs, sigma_t_sq,zL,zS,thetaE,yh])
state = sampler.run_mcmc(p0, 1000, progress=True)

sampler.reset()
sampler.run_mcmc(state, 10000, progress=True)
samples = sampler.get_chain(flat=True)

np.savetxt('emcee_H0_lambda_seed_0_ind_5081_samples.txt',samples)

label = [r'$H_0$',r'$\lambda$']#,r'$\Omega_{m0}$']
fig = corner.corner(samples, labels=label, bins=50, smooth=2, label_kwargs=dict(fontsize=12), color='#0072C1', levels=(1-np.exp(-0.5),1-np.exp(-2),1-np.exp(-4.5)), quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
plt.savefig('../plots/emcee_H0_lambda_seed_0_ind_5081.png')
