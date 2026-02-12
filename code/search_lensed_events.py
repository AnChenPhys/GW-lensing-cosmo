import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.interpolate import interp1d
from scipy.integrate import simps
import scipy.constants as constants
from lal import MTSUN_SI, PC_SI
import pickle
from astropy.cosmology import FlatLambdaCDM
import random
from optparse import OptionParser

# parser = OptionParser()
# parser.add_option('--seed', type='int', dest='seed')
# (options, args) = parser.parse_args()
# seed = options.seed

def mu_plus(y):
    return 1+1/y
def mu_minus(y):
    return -1+1/y
def theta_plus(y,thetaE):
    return (y+1)*thetaE
def theta_minus(y,thetaE):
    return (y-1)*thetaE
def t_delay_geom_plus(y):
    return -y-0.5
def t_delay_geom_minus(y):
    return y-0.5
def DeltaT(y):
    return t_delay_geom_minus(y)-t_delay_geom_plus(y)

Om0 = 0.308
H0 = 67.7
astropy_cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)

n_star = 8e-3*0.68**3  #h^3/Mpc^3
sigma_star = 144000
alpha = 2.49
beta = 2.29
Tobs = 5*365*24*3600
F_star = 16*np.pi**3*n_star*(sigma_star/constants.c)**4 * gamma((4+alpha)/beta)/gamma(alpha/beta)

zs_arr = np.logspace(-2,1,1000)
DS = astropy_cosmo.angular_diameter_distance(zs_arr).value
delta_t_star = 32*np.pi**2*(sigma_star/constants.c)**4 * DS*1e6*PC_SI/constants.c*(1+zs_arr)

tau = F_star/30*DS**3*(1+zs_arr)**3 * (1-gamma((8+alpha)/beta)/gamma((4+alpha)/beta)/7*delta_t_star/Tobs)
tau_interp = interp1d(zs_arr,tau)

M_star = 10**10.79
phi_star_1 = 10**(-3.31)#*0.7**3
alpha_s_1 = -1.69
phi_star_2 = 10**(-2.01)#*0.7**3
alpha_s_2 = -0.79

ML_arr = np.logspace(9,11.5,10000)
p_ML = phi_star_1 * np.exp(-ML_arr/M_star) * (ML_arr/M_star)**(alpha_s_1+1) + phi_star_2 * np.exp(-ML_arr/M_star) * (ML_arr/M_star)**(alpha_s_2+1)

fr = open("/home/ansonchen/dark_energy_O4a/O6_Tobs5_snr11/GW_injections_O6.p", "rb")
data = pickle.load(fr)
fr.close()

ind_good_num = 0
seed_list = []
lensed_index_list = []
y_ind_list = []
ML_ind_list = []
zL_ind_list = []
zS_ind_list = []
for seed in range(10):
    print('\n')
    np.random.seed(seed)
    samples = np.zeros(len(data['injections_parameters']['zs']))
    for i in range(len(data['injections_parameters']['zs'])):
        zs = data['injections_parameters']['zs'][i]
        if zs<3:
            samples[i] = np.random.choice([0,1], size=1, p=[1-tau_interp(zs),tau_interp(zs)])
    lensed_index = np.where(samples>0)[0]
    print('lensed index', lensed_index)

    yarr=np.linspace(0,1,10000)
    random.seed(1000+seed)
    y_ind = np.zeros(len(lensed_index))
    for i in range(len(lensed_index)):
        y_ind[i] = random.choices(yarr,weights=yarr)[0]
    print('y', y_ind)

    random.seed(2000+seed)
    ML_ind = np.zeros(len(lensed_index))
    for i in range(len(lensed_index)):
        ML_ind[i] = random.choices(ML_arr,weights=p_ML)[0] *10
    print('ML', ML_ind)

    random.seed(3000+seed)
    zL_ind = np.zeros(len(lensed_index))
    for i in range(len(lensed_index)):
        zL_arr = np.linspace(0.01,data['injections_parameters']['zs'][lensed_index][i],10000)
        zL_ind[i] = random.choices(zL_arr,weights=astropy_cosmo.differential_comoving_volume(zL_arr).value)[0]
    print('zL', zL_ind)
    print('zS', data['injections_parameters']['zs'][lensed_index])

    DL = astropy_cosmo.angular_diameter_distance(zL_ind).value
    DS = astropy_cosmo.angular_diameter_distance(data['injections_parameters']['zs'][lensed_index]).value
    DLS = astropy_cosmo.angular_diameter_distance_z1z2(zL_ind,data['injections_parameters']['zs'][lensed_index]).value
    thetaE = np.sqrt(4*np.pi*ML_ind*MTSUN_SI*constants.c/PC_SI *DLS/DS/DL/1e6)
    # print('theta_E', np.rad2deg(thetaE)*3600) #arcsec
    theta_p = theta_plus(y_ind,thetaE)
    theta_m = theta_minus(y_ind,thetaE)
    print('theta',np.rad2deg(theta_p)*3600,np.rad2deg(theta_m)*3600)

    print(np.where(np.rad2deg(np.abs(theta_m))*3600>0.2)[0])
    print(np.array(data['injections_parameters']['snrs'])[lensed_index])
    print(np.where(np.array(data['injections_parameters']['snrs'])[lensed_index]*np.sqrt(np.array(mu_minus(y_ind)))>4)[0])
    ind_good = np.intersect1d( np.where(np.rad2deg(np.abs(theta_m))*3600>0.2)[0], np.where(np.array(data['injections_parameters']['snrs'])[lensed_index]*np.sqrt(np.array(mu_minus(y_ind)))>4)[0] )
    print('ind_good',ind_good)

    if len(ind_good)>0:
        ind_good_num += 1
        seed_list.append(seed)
        lensed_index_list.append(lensed_index[ind_good])
        y_ind_list.append(y_ind[ind_good])
        ML_ind_list.append(ML_ind[ind_good])
        zL_ind_list.append(zL_ind[ind_good])
        zS_ind_list.append(data['injections_parameters']['zs'][lensed_index][ind_good])

ind_dict = dict(seed=seed_list,lensed_index=lensed_index_list,y_ind=y_ind_list,ML_ind=ML_ind_list,zL_ind=zL_ind_list,zS_ind=zS_ind_list)
np.save('event_ind_O6.npy',ind_dict)
print(ind_good_num)