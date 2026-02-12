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

def log_likelihood(free_param,dL_mean,dL_std,t_delay_obs,zL,zS,thetaE,y_mean,y_std):
    H0 = free_param[0]
    Om0 = free_param[1]
    
    loglik = 0
    if 10<=H0<=200 and 0.01<=Om0<=0.99:
        cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)
        for i in range(len(dL_mean)):
            loglik += -0.5 * (dL_mean[i] - cosmo.luminosity_distance(zS[i]).value)**2/dL_std[i]**2

            t_delay_thy = t_delay(H0,Om0,zL[i],zS[i],thetaE[i],y_mean[i])
            delta_t_delay = t_delay(H0,Om0,zL[i],zS[i],thetaE[i],y_std[i])
            loglik += -0.5 * (t_delay_obs[i] - t_delay_thy)**2/(delta_t_delay)**2
    else:
        loglik = -np.inf

    return loglik

ind_dict = np.load('event_ind.npy',allow_pickle=True).item()
seed_arr = []

for i in range(len(ind_dict['seed'])):
# for i in range(n*90,(n+1)*90):
#     if i >= len(ind_dict['seed']):
#         break

    seed = ind_dict['seed'][i]
    np.random.seed(seed)
    select_arr = []
    for k in range(len(ind_dict['lensed_index'][i])):
        select_arr.append(np.random.choice([0,1], size=1, p=[1-0.43,0.43])[0])
    visible_ind = np.where(ind_dict['lensed_index'][i] * np.array(select_arr)>0)[0]
    print(visible_ind)
    if np.size(visible_ind)==0:
        continue

    print(ind_dict['lensed_index'][i][visible_ind])
    seed_arr.append(seed)
    
    lensed_index = ind_dict['lensed_index'][i][visible_ind] #[2836,5081,8163]
    ML_ind = ind_dict['ML_ind'][i][visible_ind] #7.43e11 #5.34e10 #2.17e11
    y_ind = ind_dict['y_ind'][i][visible_ind] #0.818 #0.315 #0.594
    zL_ind = ind_dict['zL_ind'][i][visible_ind] #0.436 #0.607 #1.033
    zS_ind = ind_dict['zS_ind'][i][visible_ind] #0.688 #1.044 #1.272

    dL_mean = np.zeros(len(lensed_index))
    dL_std = np.zeros(len(lensed_index))
    y_mean = np.zeros(len(lensed_index))
    y_std = np.zeros(len(lensed_index))

    for n in range(len(lensed_index)):
        ind = lensed_index[n]
        # ML = ML_ind[n]
        # yh = y_ind[n]
        # zL = zL_ind[n]
        # zS = zS_ind[n]

        with open(f'../samples_y/sampling_sis_GO_seed_{seed}_ind_{ind}/label_result.json', 'r') as file: 
            data = json.load(file)

        dL_mean[n] = np.mean(data['posterior']['content']['DL'])
        dL_std[n] = np.std(data['posterior']['content']['DL'])
        y_mean[n] = np.mean(data['posterior']['content']['y'])
        y_std[n] = np.std(data['posterior']['content']['y'])

    t_delay_obs = DeltaT(y_ind)*4*np.pi*ML_ind*(1+zL_ind)*MTSUN_SI

    H0 = 67.7
    Om0 = 0.308
    cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)

    DL = cosmo.angular_diameter_distance(zL_ind).value
    DS = cosmo.angular_diameter_distance(zS_ind).value
    DLS = cosmo.angular_diameter_distance_z1z2(zL_ind,zS_ind).value
    thetaE = np.sqrt(4*np.pi*ML_ind*MTSUN_SI*constants.c/PC_SI *DLS/DS/DL/1e6)

    nwalkers = 32
    ndim = 2
    p0 = np.random.rand(nwalkers, ndim)*np.array([190,0.9]) + np.array([10,0.05])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=[dL_mean, dL_std, t_delay_obs, zL_ind,zS_ind,thetaE,y_mean, y_std])
    state = sampler.run_mcmc(p0, 100, progress=True)

    sampler.reset()
    sampler.run_mcmc(state, 10000, progress=True)
    samples = sampler.get_chain(flat=True)

    np.savetxt(f'emcee_H0_Om0_seed_{seed}_samples_y.txt',samples)
    np.savetxt('event_ind_seed.txt',seed_arr)

    label = [r'$H_0$',r'$\Omega_{m0}$']
    fig = corner.corner(samples, labels=label, bins=50, smooth=2, label_kwargs=dict(fontsize=12), color='#0072C1', levels=(1-np.exp(-0.5),1-np.exp(-2),1-np.exp(-4.5)), quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(f'../plots/emcee_H0_Om0_seed_{seed}_y.png')
