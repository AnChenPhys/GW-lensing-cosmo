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
plt.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True

# %config InlineBackend.figure_format = 'retina'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ind_dict = np.load('event_ind.npy',allow_pickle=True).item()

H0 = 67.7
Om0 = 0.308

H0_arr = np.linspace(20,120,200)
Om0_arr = np.linspace(0.01,0.99,100)
param_values = [H0_arr,Om0_arr]

H0_arr_XG = np.linspace(60,80,200)
Om0_arr_XG = np.linspace(0.2,0.4,100)
param_values_XG = [H0_arr_XG,Om0_arr_XG]

H0_lim_low, H0_lim_high = 45, 90
Om0_lim_low, Om0_lim_high = 0.01, 0.9
lim_low = [H0_lim_low,Om0_lim_low]
lim_high = [H0_lim_high,Om0_lim_high]

ndim = 2
label = [r'$H_0$',r'$\Omega_{m0}$']
fig, ax = plt.subplots(ndim, ndim, figsize=[6,6],constrained_layout=True)

for column in np.arange(0,ndim):
    for row in np.arange(0,ndim):
        indices = list(range(ndim))
        if column > row:
            if row!=0 or column!=1:
                fig.delaxes(ax[row][column])

        elif row == column:
            indices.remove(row)
            
            # ind_lik_norm_avg = 0
            # for i in range(1): #range(len(ind_dict['seed'])):
            #     seed = ind_dict['seed'][i]
            #     emcee_samples = np.loadtxt(f'../emcee_samples/emcee_H0_Om0_seed_{seed}_samples.txt')

            #     p1d_kde = gaussian_kde(emcee_samples[:,row])
            #     ind_lik_norm_avg += p1d_kde(param_values[row]) #/ len(ind_dict['seed'])

            ind_lik_norm_avg = np.loadtxt(f'average_lik_y_data/average_lik_1d_{row}.txt')
            ind_lik_norm_avg_O6 = np.loadtxt(f'average_lik_y_data/average_lik_1d_{row}_O6.txt')
            ind_lik_norm_avg_XG = np.loadtxt(f'average_lik_y_data/average_lik_1d_{row}_XG.txt')

            ax[row,column].plot(param_values[row],ind_lik_norm_avg/max(ind_lik_norm_avg),linewidth=2)#,color='b')
            ax[row,column].plot(param_values[row],ind_lik_norm_avg_O6/max(ind_lik_norm_avg_O6),linewidth=2)#,color='orange')
            ax[row,column].plot(param_values_XG[row],ind_lik_norm_avg_XG/max(ind_lik_norm_avg_XG),linewidth=2)#,color='g')
            ax[row,column].set_xlim(lim_low[row],lim_high[row])
            ax[row,column].set_ylim(0,1.2)

            con_int = confidence_interval(ind_lik_norm_avg, param_values[row])
            con_int_O6 = confidence_interval(ind_lik_norm_avg_O6, param_values[row])
            con_int_XG = confidence_interval(ind_lik_norm_avg_XG, param_values_XG[row])
            # ax[row,column].text(0.5,0.95, '%.2f + %.2f - %.2f'%(con_int.map, con_int.upper_level-con_int.map, con_int.map-con_int.lower_level), fontsize=15, horizontalalignment='center',verticalalignment='center', transform=ax[row,column].transAxes)
            print('%.3f + %.3f - %.3f'%(con_int.map, con_int.upper_level-con_int.map, con_int.map-con_int.lower_level))
            print('%.3f + %.3f - %.3f'%(con_int_O6.map, con_int_O6.upper_level-con_int_O6.map, con_int_O6.map-con_int_O6.lower_level))
            print('%.3f + %.3f - %.3f'%(con_int_XG.map, con_int_XG.upper_level-con_int_XG.map, con_int_XG.map-con_int_XG.lower_level))

            ax[0,0].set_xlim(40,100)#(param_values[0][0],param_values[0][-1])
            ax[1,1].set_xlim(param_values[1][0],0.9)#param_values[1][-1])
            # ax[row,column].xaxis.set_tick_params(labelsize=12)
            # ax[row,column].yaxis.set_tick_params(labelsize=12)
            ax[row,column].set_yticks([])
            if column != 1:
                ax[row,column].set_xticks([])
            else:
                ax[row,column].set_xticks([0.2,0.5,0.8])

        else:
            indices.remove(row)
            indices.remove(column)
            comb_likelihood_avg = 0
            # for i in range(1): #range(len(ind_dict['seed'])):
            #     seed = ind_dict['seed'][i]
            #     emcee_samples = np.loadtxt(f'../emcee_samples/emcee_H0_Om0_seed_{seed}_samples.txt')
            #     p2d_kde = gaussian_kde(emcee_samples.T)

            #     p2d = np.zeros((len(H0_arr),len(Om0_arr)))
            #     for i in range(len(H0_arr)):
            #         for j in range(len(Om0_arr)):
            #             p2d[i,j] = p2d_kde((H0_arr[i], Om0_arr[j]))
            #     comb_likelihood_avg += p2d #/ len(ind_dict['seed'])

            comb_likelihood_avg = np.loadtxt(f'average_lik_y_data/average_lik_2d_new.txt')
            comb_likelihood_avg_O6 = np.loadtxt(f'average_lik_y_data/average_lik_2d_O6.txt')
            comb_likelihood_avg_XG = np.loadtxt(f'average_lik_y_data/average_lik_2d_XG.txt')
            
            ax[row,column].contourf(param_values[column],param_values[row],comb_likelihood_avg.T,levels=(np.exp(-2)*np.max(comb_likelihood_avg),np.exp(-0.5)*np.max(comb_likelihood_avg),np.max(comb_likelihood_avg)),cmap='Blues',alpha=1) #np.exp(-4.5)*np.max(comb_likelihood_avg), 
            ax[row,column].contourf(param_values[column],param_values[row],comb_likelihood_avg_O6.T,levels=(np.exp(-2)*np.max(comb_likelihood_avg_O6),np.exp(-0.5)*np.max(comb_likelihood_avg_O6),np.max(comb_likelihood_avg_O6)),cmap='Oranges',alpha=1) #np.exp(-4.5)*np.max(comb_likelihood_avg), 
            ax[row,column].contourf(param_values_XG[column],param_values_XG[row],comb_likelihood_avg_XG.T,levels=(np.exp(-2)*np.max(comb_likelihood_avg_XG),np.exp(-0.5)*np.max(comb_likelihood_avg_XG),np.max(comb_likelihood_avg_XG)),cmap='Greens',alpha=1) #np.exp(-4.5)*np.max(comb_likelihood_avg), 
            ax[row,column].set_xlim(lim_low[column],lim_high[column])
            ax[row,column].set_ylim(param_values[1][0],0.9)#param_values[1][-1])
            # ax[row,column].xaxis.set_tick_params(labelsize=12)
            # ax[row,column].yaxis.set_tick_params(labelsize=12)

        ax[1,0].set_yticks([0.2,0.5,0.8])
        ax[1,0].set_xticks([50,70,90])
        # ax[0,0].set_xlabel(label[0], fontsize=13)
        # ax[0,0].set_ylabel('p('+label[0]+')', fontsize=13)
        ax[1,1].set_xlabel(label[1])
        # ax[1,1].set_ylabel('p('+label[1]+')', fontsize=13)
        # ax[1,1].yaxis.set_label_position("right")
        # ax[1,1].yaxis.tick_right()
        ax[1,0].set_xlabel(label[0])
        ax[1,0].set_ylabel(label[1])

        # if column == 0:
        #     ax[row,column].set_ylabel(label[row], fontsize=16)
        # if row == ndim-1:
        #     ax[row,column].set_xlabel(label[column],fontsize=16)

        ax[0,0].axvline(H0,color='grey',linestyle='--')
        ax[1,1].axvline(Om0,color='grey',linestyle='--')

ax[0,1].set_frame_on(False)
ax[0,1].plot(0.1, 0.1, label='LVK O5')
ax[0,1].plot(0.1, 0.1, label='LVK post-O5')
ax[0,1].plot(0.1, 0.1, label='ET+CE')
ax[0,1].axis('off')
ax[0,1].legend(frameon=False, fontsize=14)

plt.savefig(f'../plots/emcee_h0_Om0_combine_averaged_new.png',dpi=600)
