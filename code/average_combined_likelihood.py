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

ind_dict = np.load('event_ind.npy',allow_pickle=True).item()

H0 = 67.7
Om0 = 0.308

h0_arr = np.linspace(20,140,1000)
Om0_arr = np.linspace(0.05,0.95,100)
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
            
            ind_lik_norm_avg = 0
            for i in range(20): #range(len(ind_dict['seed'])):
                seed = ind_dict['seed'][i]
                comb_likelihood = np.loadtxt(f'../likelihood/H0_Om0_loglik_seed_{seed}_combine.txt')

                ind_lik = np.sum(comb_likelihood,axis=tuple(indices))
                ind_lik_norm = ind_lik/np.sum(ind_lik)/(param_values[row][1]-param_values[row][0])
                ind_lik_norm_avg += ind_lik_norm / 20

            ax[row,column].plot(param_values[row],ind_lik_norm_avg,linewidth=2)
            ax[row,column].set_xlim(param_values[row][0],param_values[row][-1])
            ax[row,column].set_ylim(0,1.2*np.max(ind_lik_norm_avg))

            con_int = confidence_interval(ind_lik_norm_avg, param_values[row])
            ax[row,column].text(0.5,0.95, '%.2f + %.2f - %.2f'%(con_int.map, con_int.upper_level-con_int.map, con_int.map-con_int.lower_level), fontsize=15, horizontalalignment='center',verticalalignment='center', transform=ax[row,column].transAxes)

            ax[0,0].set_xlim(40,100)#(param_values[0][0],param_values[0][-1])
            ax[1,1].set_xlim(param_values[1][0],param_values[1][-1])
            ax[row,column].xaxis.set_tick_params(labelsize=15)
            ax[row,column].yaxis.set_tick_params(labelsize=15)

        else:
            indices.remove(row)
            indices.remove(column)
            comb_likelihood_avg = 0
            for i in range(20): #range(len(ind_dict['seed'])):
                seed = ind_dict['seed'][i]
                comb_likelihood = np.loadtxt(f'../likelihood/H0_Om0_loglik_seed_{seed}_combine.txt')
                comb_likelihood_avg += comb_likelihood / 20

            ax[row,column].contourf(param_values[column],param_values[row],np.sum(comb_likelihood_avg,axis=tuple(indices)).T,levels=(np.exp(-4.5), np.exp(-2),np.exp(-0.5),1),cmap='Blues')
            ax[row,column].set_xlim(40,100)#(param_values[0][0],param_values[0][-1])
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

        ax[0,0].axvline(H0,color='grey',linestyle='--')
        ax[1,1].axvline(Om0,color='grey',linestyle='--')

plt.savefig(f'../plots/lik_h0_Om0_combine_averaged.png')
