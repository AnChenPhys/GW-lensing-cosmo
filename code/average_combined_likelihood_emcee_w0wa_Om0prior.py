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
plt.rcParams.update({'font.size': 13})
matplotlib.rcParams['text.usetex'] = True

# %config InlineBackend.figure_format = 'retina'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ind_dict = np.load('event_ind.npy',allow_pickle=True).item()
# seed_list = np.loadtxt('event_ind_seed_w0wa.txt')

H0 = 67.7
Om0 = 0.308

# H0_arr = np.linspace(20,200,200)
# Om0_arr = np.linspace(0.01,0.99,100)
# w0_arr = np.linspace(-3,0,100)
# wa_arr = np.linspace(-5,5,100)
# param_values = [H0_arr,Om0_arr,w0_arr,wa_arr]

H0_arr_XG = np.linspace(40,100,200)
Om0_arr_XG = np.linspace(0.3,0.32,100)
w0_arr_XG = np.linspace(-3,0,100)
wa_arr_XG = np.linspace(-5,5,100)
param_values_XG = [H0_arr_XG,Om0_arr_XG,w0_arr_XG,wa_arr_XG]

# H0_arr_2d = np.linspace(20,200,100)
# Om0_arr_2d = np.linspace(0.01,0.99,100)
# param_values_2d = [H0_arr_2d,Om0_arr_2d,w0_arr,wa_arr]

H0_arr_XG_2d = np.linspace(40,100,100)
param_values_XG_2d = [H0_arr_XG_2d,Om0_arr_XG,w0_arr_XG,wa_arr_XG]

H0_lim_low, H0_lim_high = 40, 100
Om0_lim_low, Om0_lim_high = 0.30, 0.32
w0_lim_low, w0_lim_high = -2.8, 0
wa_lim_low, wa_lim_high = -4, 3
lim_low = [H0_lim_low,Om0_lim_low,w0_lim_low,wa_lim_low]
lim_high = [H0_lim_high,Om0_lim_high,w0_lim_high,wa_lim_high]

ndim = 4
label = [r'$H_0$',r'$\Omega_{m0}$',r'$w_0$',r'$w_a$']
label_p = [r'p($H_0$)',r'p($\Omega_{m0}$)',r'p($w_0$)',r'p($w_a$)']
fig, ax = plt.subplots(ndim, ndim, figsize=[6,6],constrained_layout=True)

# for i in range(ndim):
#     ind_lik_norm_avg = 0
#     for s in range(len(ind_dict['seed'])):
#         seed = ind_dict['seed'][s]
#         emcee_samples = np.loadtxt(f'../emcee_samples/emcee_H0_Om0_w0wa_seed_{seed}_samples.txt')

#         p1d_kde = gaussian_kde(emcee_samples[:,i])
#         ind_lik_norm_avg += p1d_kde(param_values[i]) / len(ind_dict['seed'])

#     # ind_lik_norm_avg = np.loadtxt(f'average_w0wa_lik_1d_{i}_new.txt')

#     plt.figure()
#     plt.plot(param_values[i],ind_lik_norm_avg,linewidth=2)
#     plt.xlim(param_values[i][0],param_values[i][-1])
#     plt.ylim(0,1.2*np.max(ind_lik_norm_avg))

#     # con_int = confidence_interval(ind_lik_norm_avg, param_values[i])
#     # plt.text(0.5,0.95, '%.2f + %.2f - %.2f'%(con_int.map, con_int.upper_level-con_int.map, con_int.map-con_int.lower_level), fontsize=15, horizontalalignment='center',verticalalignment='center', transform=ax[row,column].transAxes)

#     plt.xlabel(label[i])
#     plt.ylabel(label_p[i])
#     plt.savefig(f'../plots/emcee_H0_Om0_w0wa_combined_averaged_{i}_new.png')



for column in np.arange(0,ndim):
    for row in np.arange(0,ndim):
        indices = list(range(ndim))
        if column > row:
            if row!=0 or column!=3:
                fig.delaxes(ax[row][column])

        elif row == column:
            indices.remove(row)
            
            # ind_lik_norm_avg = 0
            # for i in range(len(seed_list)):
            #     seed = int(seed_list[i])
            #     emcee_samples = np.loadtxt(f'../emcee_samples_new/emcee_H0_Om0_w0wa_seed_{seed}_samples.txt')

            #     p1d_kde = gaussian_kde(emcee_samples[:,row])
            #     ind_lik_norm_avg += p1d_kde(param_values[row]) / len(seed_list)
            
            # np.savetxt(f'average_lik_w0wa_1d_{row}_new.txt',ind_lik_norm_avg)

            # ind_lik_norm_avg = np.loadtxt(f'average_lik_y_data/average_lik_w0wa_1d_{row}.txt')
            # ind_lik_norm_avg_O6 = np.loadtxt(f'average_lik_y_data/average_lik_w0wa_1d_{row}_O6.txt')
            ind_lik_norm_avg_XG = np.loadtxt(f'average_lik_y_data/average_lik_w0wa_1d_{row}_XG_Om0prior.txt')

            # ax[row,column].plot(param_values[row],ind_lik_norm_avg/max(ind_lik_norm_avg),linewidth=1.5)
            # ax[row,column].plot(param_values[row],ind_lik_norm_avg_O6/max(ind_lik_norm_avg_O6),linewidth=1.5)
            ax[row,column].plot(param_values_XG[row],ind_lik_norm_avg_XG/max(ind_lik_norm_avg_XG),linewidth=1.5)
            ax[row,column].set_xlim(lim_low[row],lim_high[row])
            ax[row,column].set_ylim(0,1.2)

            # con_int = confidence_interval(ind_lik_norm_avg, param_values[row]+10)
            # con_int_O6 = confidence_interval(ind_lik_norm_avg_O6, param_values[row]+10)
            con_int_XG = confidence_interval(ind_lik_norm_avg_XG, param_values_XG[row]+10)
            # print('%.3f + %.3f - %.3f'%(con_int.map, con_int.upper_level-con_int.map, con_int.map-con_int.lower_level))
            # print('%.3f + %.3f - %.3f'%(con_int_O6.map, con_int_O6.upper_level-con_int_O6.map, con_int_O6.map-con_int_O6.lower_level))
            print('%.3f + %.3f - %.3f'%(con_int_XG.map, con_int_XG.upper_level-con_int_XG.map, con_int_XG.map-con_int_XG.lower_level))
            # ax[row,column].text(0.5,0.95, '%.2f + %.2f - %.2f'%(con_int.map, con_int.upper_level-con_int.map, con_int.map-con_int.lower_level), fontsize=15, horizontalalignment='center',verticalalignment='center', transform=ax[row,column].transAxes)
            # ax[row,column].xaxis.set_tick_params(labelsize=15)
            # ax[row,column].yaxis.set_tick_params(labelsize=15)
            ax[row,column].set_yticks([])
            if row != 3:
                ax[row,column].set_xticks([])
            else:
                ax[row,column].set_xticks([-3,0,3])

        else:
            indices.remove(row)
            indices.remove(column)
            # comb_likelihood_avg = 0
            # for i in range(len(seed_list)):
            #     seed = int(seed_list[i])
            #     emcee_samples = np.loadtxt(f'../emcee_samples_new/emcee_H0_Om0_w0wa_seed_{seed}_samples.txt')
            #     p2d_kde = gaussian_kde([emcee_samples.T[row],emcee_samples.T[column]])

            #     p2d = np.zeros((len(param_values_2d[row]),len(param_values_2d[column])))
            #     for j in range(len(param_values_2d[row])):
            #         for k in range(len(param_values_2d[column])):
            #             p2d[j,k] = p2d_kde((param_values_2d[row][j], param_values_2d[column][k]))
            #     comb_likelihood_avg += p2d / len(seed_list)

            # np.savetxt(f'average_lik_w0wa_2d_{row}_{column}_new.txt',comb_likelihood_avg)

            # comb_likelihood_avg = np.loadtxt(f'average_lik_y_data/average_lik_w0wa_2d_{row}_{column}.txt')
            # comb_likelihood_avg_O6 = np.loadtxt(f'average_lik_y_data/average_lik_w0wa_2d_{row}_{column}_O6.txt')
            comb_likelihood_avg_XG = np.loadtxt(f'average_lik_y_data/average_lik_w0wa_2d_{row}_{column}_XG_Om0prior.txt')
            
            # ax[row,column].contourf(param_values_2d[column],param_values_2d[row],comb_likelihood_avg,levels=(np.exp(-2)*np.max(comb_likelihood_avg),np.exp(-0.5)*np.max(comb_likelihood_avg),np.max(comb_likelihood_avg)),cmap='Blues')
            # ax[row,column].contourf(param_values_2d[column],param_values_2d[row],comb_likelihood_avg_O6,levels=(np.exp(-2)*np.max(comb_likelihood_avg_O6),np.exp(-0.5)*np.max(comb_likelihood_avg_O6),np.max(comb_likelihood_avg_O6)),cmap='Oranges')
            ax[row,column].contourf(param_values_XG_2d[column],param_values_XG_2d[row],comb_likelihood_avg_XG,levels=(np.exp(-2)*np.max(comb_likelihood_avg_XG),np.exp(-0.5)*np.max(comb_likelihood_avg_XG),np.max(comb_likelihood_avg_XG)),cmap='Greens')
            ax[row,column].set_xlim(lim_low[column],lim_high[column])
            ax[row,column].set_ylim(lim_low[row],lim_high[row])
            # ax[row,column].xaxis.set_tick_params(labelsize=15)
            # ax[row,column].yaxis.set_tick_params(labelsize=15)

        # ax[0,0].set_xlabel(label[0], fontsize=16)
        # ax[0,0].set_ylabel('p('+label[0]+')', fontsize=16)
        for i in range(4):
            ax[3,i].set_xlabel(label[i], fontsize=15)
        for i in range(1,4):
            ax[i,0].set_ylabel(label[i], fontsize=15)

        ax[3,0].set_xticks([40,70,100])
        ax[3,1].set_xticks([0.2,0.5,0.8])
        ax[3,2].set_xticks([-2,-1,0])
        ax[1,0].set_yticks([0.2,0.5,0.8])
        ax[2,0].set_yticks([-2,-1,0])
        ax[3,0].set_yticks([-3,0,3])
        if row != 3:
            ax[row,column].set_xticks([])
        if column != 0:
            ax[row,column].set_yticks([])

        # if column == 0:
        #     ax[row,column].set_ylabel(label[row], fontsize=16)
        # if row == ndim-1:
        #     ax[row,column].set_xlabel(label[column],fontsize=16)

        ax[0,0].axvline(H0,color='grey',linestyle='--')
        ax[1,1].axvline(Om0,color='grey',linestyle='--')
        ax[2,2].axvline(-1,color='grey',linestyle='--')
        ax[3,3].axvline(0,color='grey',linestyle='--')

ax[0,3].set_frame_on(False)
ax[0,3].plot(0.1, 0.1, label='LVK O5')
ax[0,3].plot(0.1, 0.1, label='LVK post-O5')
ax[0,3].plot(0.1, 0.1, label='ET+CE')
ax[0,3].axis('off')
ax[0,3].legend(frameon=False, fontsize=8)

plt.savefig(f'../plots/emcee_h0_Om0_w0wa_combine_averaged_Om0prior.png',dpi=600)
