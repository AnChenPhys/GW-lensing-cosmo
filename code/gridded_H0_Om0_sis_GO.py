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
from utils import *
# import matplotlib
# plt.rcParams.update({'font.size': 15})
# matplotlib.rcParams['text.usetex'] = True

# # %config InlineBackend.figure_format = 'retina'
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

def Loglik(free_param, dL_mean, dL_std, t_delay_obs, sigma_t_sq,zL,zS,thetaE,yh):
    loglik = 0
    H0 = free_param[0]
    Om0 = free_param[1]
    # zS = free_param[1]
    # lamb = free_param[1]

    cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)
    loglik += -0.5*(dL_mean - cosmo.luminosity_distance(zS).value)**2/dL_std**2

    # t_delay_thy = t_delay(H0,Om0,zL,zS,thetaE,yh)
    # loglik += -0.5*(t_delay_obs - t_delay_thy)**2/sigma_t_sq**2

    return loglik

ind_dict = np.load('event_ind.npy',allow_pickle=True).item()

# i = 0
for i in range(1): #range(len(ind_dict['seed'])):
    seed = ind_dict['seed'][i]
    lensed_index = ind_dict['lensed_index'][i] #[2836,5081,8163]
    ML_ind = ind_dict['ML_ind'][i] #7.43e11 #5.34e10 #2.17e11
    y_ind = ind_dict['y_ind'][i] #0.818 #0.315 #0.594
    zL_ind = ind_dict['zL_ind'][i] #0.436 #0.607 #1.033
    zS_ind = ind_dict['zS_ind'][i] #0.688 #1.044 #1.272

    for n in range(len(lensed_index)):
        ind = lensed_index[n]
        ML = ML_ind[n]
        yh = y_ind[n]
        zL = zL_ind[n]
        zS = zS_ind[n]

        with open(f'../samples/sampling_sis_GO_seed_{seed}_ind_{ind}/label_result.json', 'r') as file: 
            data = json.load(file)

        tc_plus = np.array(data['posterior']['content']['tc_plus'])*1e-4
        tc_minus = np.array(data['posterior']['content']['tc_minus'])*1e-4
        tc_diff = np.mean(tc_minus) - np.mean(tc_plus)
        sigma_t_sq = np.std(tc_plus)**2 + np.std(tc_minus)**2

        # kde_dL = gaussian_kde(data['posterior']['content']['DL'])
        # dL_arr = np.linspace(min(data['posterior']['content']['DL']),max(data['posterior']['content']['DL']),10000)
        # p_dL_rmp = kde_dL(dL_arr)/dL_arr**2
        # dL_data_rw = random.choices(dL_arr,weights=p_dL_rmp,k=10000)
        # dL_mean = np.mean(dL_data_rw)
        # dL_std = np.std(dL_data_rw)
        dL_mean = np.mean(data['posterior']['content']['DL'])
        dL_std = np.std(data['posterior']['content']['DL'])

        t_delay_obs = DeltaT(yh)*4*np.pi*ML*(1+zL)*MTSUN_SI + tc_diff

        H0_fid = 67.7
        Om0_fid = 0.308
        cosmo_fid = FlatLambdaCDM(H0=H0_fid,Om0=Om0_fid)

        DL = cosmo_fid.angular_diameter_distance(zL).value
        DS = cosmo_fid.angular_diameter_distance(zS).value
        DLS = cosmo_fid.angular_diameter_distance_z1z2(zL,zS).value
        thetaE = np.sqrt(4*np.pi*ML*MTSUN_SI*constants.c/PC_SI *DLS/DS/DL/1e6)

        h0_arr = np.linspace(10,200,100)
        Om0_arr = np.linspace(0.05,0.95,100)
        loglik = np.zeros((len(h0_arr),len(Om0_arr)))
        comb_likelihood = np.ones((len(h0_arr),len(Om0_arr)))
        for i in range(len(h0_arr)):
            for j in range(len(Om0_arr)):
                loglik[i,j] = Loglik([h0_arr[i],Om0_arr[j]], dL_mean, dL_std, t_delay_obs, sigma_t_sq,zL,zS,thetaE,yh)
        np.savetxt(f'../likelihood/H0_Om0_loglik_seed_{seed}_ind_{ind}.txt',loglik)
        # lik = np.loadtxt('H0_lambda_loglik_seed_0_ind_5081.txt')
        comb_likelihood *= np.exp(loglik)

    np.savetxt(f'../likelihood/H0_Om0_loglik_seed_{seed}_combine.txt',comb_likelihood)
    # comb_likelihood = np.loadtxt(f'../likelihood/H0_Om0_loglik_seed_{seed}_combine.txt')

    ndim = 2
    param_values = [h0_arr,Om0_arr]
    label = [r'$H_0$',r'$\Omega_{m0}$']
    fig, ax = plt.subplots(ndim, ndim, figsize=[8,8],constrained_layout=True)

    for column in np.arange(0,ndim):
        for row in np.arange(0,ndim):
            indices = list(range(ndim))
            if column > row:
                fig.delaxes(ax[row][column])

            elif row == column:
                indices.remove(row)
                ind_lik = np.sum(comb_likelihood,axis=tuple(indices))
                ind_lik_norm = ind_lik/np.sum(ind_lik)/(param_values[row][1]-param_values[row][0])

                ax[row,column].plot(param_values[row],ind_lik_norm,linewidth=2)
                ax[row,column].set_xlim(param_values[row][0],param_values[row][-1])
                ax[row,column].set_ylim(0,1.2*np.max(ind_lik_norm))

                con_int = confidence_interval(ind_lik_norm, param_values[row])
                ax[row,column].text(0.5,0.95, '%.2f + %.2f - %.2f'%(con_int.map, con_int.upper_level-con_int.map, con_int.map-con_int.lower_level), fontsize=15, horizontalalignment='center',verticalalignment='center', transform=ax[row,column].transAxes)

                ax[0,0].set_xlim(param_values[0][0],param_values[0][-1])
                ax[1,1].set_xlim(param_values[1][0],param_values[1][-1])
                ax[row,column].xaxis.set_tick_params(labelsize=15)
                ax[row,column].yaxis.set_tick_params(labelsize=15)

            else:
                indices.remove(row)
                indices.remove(column)
                comb_likelihood_2d = np.sum(comb_likelihood,axis=tuple(indices)).T
                ax[row,column].contourf(param_values[column],param_values[row],comb_likelihood_2d,levels=(np.exp(-4.5), np.exp(-2),np.exp(-0.5),1),cmap='Blues')
                ax[row,column].set_xlim(param_values[0][0],param_values[0][-1])
                ax[row,column].set_ylim(param_values[1][0],param_values[1][-1])
                ax[row,column].xaxis.set_tick_params(labelsize=15)
                ax[row,column].yaxis.set_tick_params(labelsize=15)

            ax[0,0].set_xlabel(label[0], fontsize=16)
            ax[0,0].set_ylabel('p('+label[0]+')', fontsize=16)
            ax[1,1].set_xlabel(label[1], fontsize=16)
            ax[1,1].set_ylabel('p('+label[1]+')', fontsize=16)
            ax[1,1].yaxis.set_label_position("right")
            ax[1,1].yaxis.tick_right()
            ax[1,0].set_xlabel(label[0], fontsize=16)
            ax[1,0].set_ylabel(label[1], fontsize=16)

            # if column == 0:
            #     ax[row,column].set_ylabel(label[row], fontsize=16)
            # if row == ndim-1:
            #     ax[row,column].set_xlabel(label[column],fontsize=16)

            ax[0,0].axvline(H0_fid,color='grey',linestyle='--')
            ax[1,1].axvline(Om0_fid,color='grey',linestyle='--')

    plt.savefig(f'../plots/lik_h0_Om0_combine_seed_{seed}.png')
