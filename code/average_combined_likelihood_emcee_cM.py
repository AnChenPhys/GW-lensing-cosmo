import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
import scipy.constants as constants
from scipy.stats import gaussian_kde
from lal import MTSUN_SI, PC_SI
from astropy.cosmology import FlatLambdaCDM
import json
from utils import *
import matplotlib
plt.rcParams.update({'font.size': 15})
matplotlib.rcParams['text.usetex'] = True

# %config InlineBackend.figure_format = 'retina'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ind_dict = np.load('event_ind.npy',allow_pickle=True).item()

H0 = 67.7
Om0 = 0.308

H0_arr = np.linspace(20,140,200)
Om0_arr = np.linspace(0.01,0.99,100)
cM_arr = np.linspace(-10,10,200)
param_values = [H0_arr,Om0_arr,cM_arr]

H0_arr_XG = np.linspace(55,80,200)
Om0_arr_XG = np.linspace(0.05,0.8,200)
cM_arr_XG = np.linspace(-2,2,200)
param_values_XG = [H0_arr_XG,Om0_arr_XG,cM_arr_XG]

H0_arr_2d = np.linspace(20,140,100)
Om0_arr_2d = np.linspace(0.01,0.99,100)
cM_arr_2d = np.linspace(-10,10,100)
param_values_2d = [H0_arr_2d,Om0_arr_2d,cM_arr_2d]

H0_arr_XG_2d = np.linspace(55,80,100)
Om0_arr_XG_2d = np.linspace(0.05,0.8,100)
cM_arr_XG_2d = np.linspace(-2,2,100)
param_values_XG_2d = [H0_arr_XG_2d,Om0_arr_XG_2d,cM_arr_XG_2d]

H0_lim_low, H0_lim_high = 35, 100
Om0_lim_low, Om0_lim_high = 0.01, 0.9
cM_lim_low, cM_lim_high = -4, 4
lim_low = [H0_lim_low,Om0_lim_low,cM_lim_low]
lim_high = [H0_lim_high,Om0_lim_high,cM_lim_high]

ndim = 3
label = [r'$H_0$',r'$\Omega_{m0}$',r'$c_M$']
fig, ax = plt.subplots(ndim, ndim, figsize=[6,6],constrained_layout=True)

for column in np.arange(0,ndim):
    for row in np.arange(0,ndim):
        indices = list(range(ndim))
        if column > row:
            if row!=0 or column!=2:
                fig.delaxes(ax[row][column])

        elif row == column:
            indices.remove(row)
            
            # ind_lik_norm_avg = 0
            # for i in range(1): #range(len(ind_dict['seed'])):
            #     seed = ind_dict['seed'][i]
            #     emcee_samples = np.loadtxt(f'../emcee_samples/emcee_H0_Om0_cM_seed_{seed}_samples.txt')

            #     p1d_kde = gaussian_kde(emcee_samples[:,row])
            #     ind_lik_norm_avg += p1d_kde(param_values[row]) #/ len(ind_dict['seed'])

            ind_lik_norm_avg = np.loadtxt(f'average_lik_y_data/average_lik_cM_1d_{row}.txt')
            ind_lik_norm_avg_O6 = np.loadtxt(f'average_lik_y_data/average_lik_cM_1d_{row}_O6.txt')
            ind_lik_norm_avg_XG = np.loadtxt(f'average_lik_y_data/average_lik_cM_1d_{row}_XG.txt')

            ax[row,column].plot(param_values[row],ind_lik_norm_avg/max(ind_lik_norm_avg),linewidth=1.75)
            ax[row,column].plot(param_values[row],ind_lik_norm_avg_O6/max(ind_lik_norm_avg_O6),linewidth=1.75)
            ax[row,column].plot(param_values_XG[row],ind_lik_norm_avg_XG/max(ind_lik_norm_avg_XG),linewidth=1.75)
            ax[row,column].set_xlim(lim_low[row],lim_high[row])
            ax[row,column].set_ylim(0,1.2)

            con_int = confidence_interval(ind_lik_norm_avg, param_values[row]+10)
            con_int_O6 = confidence_interval(ind_lik_norm_avg_O6, param_values[row]+10)
            con_int_XG = confidence_interval(ind_lik_norm_avg_XG, param_values_XG[row]+10)
            print('%.3f + %.3f - %.3f'%(con_int.map, con_int.upper_level-con_int.map, con_int.map-con_int.lower_level))
            print('%.3f + %.3f - %.3f'%(con_int_O6.map, con_int_O6.upper_level-con_int_O6.map, con_int_O6.map-con_int_O6.lower_level))
            print('%.3f + %.3f - %.3f'%(con_int_XG.map, con_int_XG.upper_level-con_int_XG.map, con_int_XG.map-con_int_XG.lower_level))

            ax[row,column].set_yticks([])
            if row != 2:
                ax[row,column].set_xticks([])

        else:
            indices.remove(row)
            indices.remove(column)
            comb_likelihood_avg = 0
            # for i in range(1): #range(len(ind_dict['seed'])):
            #     seed = ind_dict['seed'][i]
            #     emcee_samples = np.loadtxt(f'../emcee_samples/emcee_H0_Om0_cM_seed_{seed}_samples.txt')
            #     p2d_kde = gaussian_kde([emcee_samples.T[row],emcee_samples.T[column]])

            #     p2d = np.zeros((len(param_values_2d[row]),len(param_values_2d[column])))
            #     for j in range(len(param_values_2d[row])):
            #         for k in range(len(param_values_2d[column])):
            #             p2d[j,k] = p2d_kde((param_values_2d[row][j], param_values_2d[column][k]))
            #     comb_likelihood_avg += p2d #/ len(ind_dict['seed'])

            comb_likelihood_avg = np.loadtxt(f'average_lik_y_data/average_lik_cM_2d_{row}_{column}.txt')
            comb_likelihood_avg_O6 = np.loadtxt(f'average_lik_y_data/average_lik_cM_2d_{row}_{column}_O6.txt')
            comb_likelihood_avg_XG = np.loadtxt(f'average_lik_y_data/average_lik_cM_2d_{row}_{column}_XG.txt')
            
            ax[row,column].contourf(param_values_2d[column],param_values_2d[row],comb_likelihood_avg,levels=(np.exp(-2)*np.max(comb_likelihood_avg),np.exp(-0.5)*np.max(comb_likelihood_avg),np.max(comb_likelihood_avg)),cmap='Blues') #np.exp(-4.5)*np.max(comb_likelihood_avg), 
            ax[row,column].contourf(param_values_2d[column],param_values_2d[row],comb_likelihood_avg_O6,levels=(np.exp(-2)*np.max(comb_likelihood_avg_O6),np.exp(-0.5)*np.max(comb_likelihood_avg_O6),np.max(comb_likelihood_avg_O6)),cmap='Oranges') #np.exp(-4.5)*np.max(comb_likelihood_avg), 
            ax[row,column].contourf(param_values_XG_2d[column],param_values_XG_2d[row],comb_likelihood_avg_XG,levels=(np.exp(-2)*np.max(comb_likelihood_avg_XG),np.exp(-0.5)*np.max(comb_likelihood_avg_XG),np.max(comb_likelihood_avg_XG)),cmap='Greens') #np.exp(-4.5)*np.max(comb_likelihood_avg), 
            ax[row,column].set_xlim(lim_low[column],lim_high[column])
            ax[row,column].set_ylim(lim_low[row],lim_high[row])

            if column != 0:
                ax[row,column].set_yticks([])
            if row != 2:
                ax[row,column].set_xticks([])

        ax[1,0].set_ylabel(label[1])
        ax[2,0].set_xlabel(label[0])
        ax[2,0].set_ylabel(label[2])
        ax[2,1].set_xlabel(label[1])
        ax[2,2].set_xlabel(label[2])

        ax[2,1].set_xticks([0.2,0.5,0.8])
        ax[1,0].set_yticks([0.2,0.5,0.8])
        ax[2,0].set_xticks([50,70,90])
        ax[2,0].set_yticks([-3,0,3])
        ax[2,2].set_xticks([-3,0,3])

        ax[0,0].axvline(H0,color='grey',linestyle='--')
        ax[1,1].axvline(Om0,color='grey',linestyle='--')
        ax[2,2].axvline(0,color='grey',linestyle='--')

ax[0,2].set_frame_on(False)
ax[0,2].plot(0.1, 0.1, label='LVK O5')
ax[0,2].plot(0.1, 0.1, label='LVK post-O5')
ax[0,2].plot(0.1, 0.1, label='ET+CE')
ax[0,2].axis('off')
ax[0,2].legend(frameon=False, fontsize=12)

plt.savefig(f'../plots/emcee_h0_Om0_cM_combine_averaged.png',dpi=600)
