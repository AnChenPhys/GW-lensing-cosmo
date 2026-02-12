import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
import scipy.constants as constants
from lal import MTSUN_SI, PC_SI
from astropy.cosmology import FlatLambdaCDM, w0waCDM
import emcee
import json
import corner
import matplotlib
plt.rcParams.update({'font.size': 15})

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--n', type='int', dest='n')
(options, args) = parser.parse_args()
n = options.n

# matplotlib.rcParams['text.usetex'] = True

# %config InlineBackend.figure_format = 'retina'
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

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

def dGW_dL_ratio_cM(z,cM,Om0):
    OL0 = 1-Om0
    return np.exp(cM/(2*OL0) * np.log((1+z)/np.power(Om0*np.power(1.+z,3.)+OL0,1./3)))

def log_likelihood(free_param,dL_mean,dL_std,t_delay_obs,sigma_t_sq,zL,zS,thetaE,yh):
    H0 = free_param[0]
    Om0 = free_param[1]
    cM = free_param[2]
    
    loglik = 0
    if 10<=H0<=200 and 0.01<=Om0<=0.99 and -10<=cM<=10:
        cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)
        for i in range(len(dL_mean)):
            dGW_dL_ratio = dGW_dL_ratio_cM(zS[i],cM,Om0)
            loglik += -0.5 * (dL_mean[i] - dGW_dL_ratio*cosmo.luminosity_distance(zS[i]).value)**2/dL_std[i]**2

            t_delay_thy = t_delay(H0,Om0,zL[i],zS[i],thetaE[i],yh[i])
            loglik += -0.5 * (t_delay_obs[i] - t_delay_thy)**2/(0.01*t_delay_obs[i])**2
    else:
        loglik = -np.inf

    return loglik

ind_dict = np.load('event_ind.npy',allow_pickle=True).item()

for i in range(1): #range(len(ind_dict['seed'])):
# for i in range(n*90,(n+1)*90):
    # if i >= len(ind_dict['seed']):
    #     break

    seed = ind_dict['seed'][i]
    lensed_index = ind_dict['lensed_index'][i] #[2836,5081,8163]
    ML_ind = ind_dict['ML_ind'][i] #7.43e11 #5.34e10 #2.17e11
    y_ind = ind_dict['y_ind'][i] #0.818 #0.315 #0.594
    zL_ind = ind_dict['zL_ind'][i] #0.436 #0.607 #1.033
    zS_ind = ind_dict['zS_ind'][i] #0.688 #1.044 #1.272

    dL_mean = np.zeros(len(lensed_index))
    dL_std = np.zeros(len(lensed_index))
    sigma_t_sq = np.zeros(len(lensed_index))
    tc_diff = np.zeros(len(lensed_index))

    for j in range(len(lensed_index)):
        ind = lensed_index[j]
        # ML = ML_ind[n]
        # yh = y_ind[n]
        # zL = zL_ind[n]
        # zS = zS_ind[n]

        with open(f'../samples/sampling_sis_GO_seed_{seed}_ind_{ind}/label_result.json', 'r') as file: 
            data = json.load(file)

        tc_plus = np.array(data['posterior']['content']['tc_plus'])*1e-4
        tc_minus = np.array(data['posterior']['content']['tc_minus'])*1e-4
        tc_diff[j] = np.mean(tc_minus) - np.mean(tc_plus)
        sigma_t_sq[j] = np.std(tc_plus)**2 + np.std(tc_minus)**2

        dL_mean[j] = np.mean(data['posterior']['content']['DL'])
        dL_std[j] = np.std(data['posterior']['content']['DL'])

    t_delay_obs = DeltaT(y_ind)*4*np.pi*ML_ind*(1+zL_ind)*MTSUN_SI + tc_diff

    H0 = 67.7
    Om0 = 0.308
    cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)

    DL = cosmo.angular_diameter_distance(zL_ind).value
    DS = cosmo.angular_diameter_distance(zS_ind).value
    DLS = cosmo.angular_diameter_distance_z1z2(zL_ind,zS_ind).value
    thetaE = np.sqrt(4*np.pi*ML_ind*MTSUN_SI*constants.c/PC_SI *DLS/DS/DL/1e6)

    nwalkers = 32
    ndim = 3
    p0 = np.random.rand(nwalkers, ndim)*np.array([190,0.9,20]) + np.array([10,0.05,-10])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=[dL_mean, dL_std, t_delay_obs, sigma_t_sq,zL_ind,zS_ind,thetaE,y_ind])
    state = sampler.run_mcmc(p0, 100, progress=True)

    sampler.reset()
    sampler.run_mcmc(state, 10000, progress=True)
    samples = sampler.get_chain(flat=True)

    np.savetxt(f'emcee_H0_Om0_cM_seed_{seed}_samples.txt',samples)

    label = [r'$H_0$',r'$\Omega_{m0}$',r'$c_M$']
    fig = corner.corner(samples, labels=label, bins=50, smooth=2, label_kwargs=dict(fontsize=12), color='#0072C1', levels=(1-np.exp(-0.5),1-np.exp(-2),1-np.exp(-4.5)), quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(f'../plots/emcee_H0_Om0_cM_seed_{seed}.png')
