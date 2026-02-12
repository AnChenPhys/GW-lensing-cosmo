# import icarogw
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.interpolate import interp1d
from scipy.integrate import simps
import scipy.constants as constants
from lal import MTSUN_SI, PC_SI
import pickle
from astropy.cosmology import FlatLambdaCDM
import bilby
import random
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--seed', type='int', dest='seed')
(options, args) = parser.parse_args()
seed = options.seed

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

fr = open("/home/ansonchen/dark_energy_O4a/O5_Tobs5_snr11/GW_injections_O5.p", "rb")
data = pickle.load(fr)
fr.close()

np.random.seed(seed)
samples = np.zeros(len(data['injections_parameters']['zs']))
for i in range(len(data['injections_parameters']['zs'])):
    zs = data['injections_parameters']['zs'][i]
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
print('theta_E', np.rad2deg(thetaE)*3600) #arcsec
theta_p = theta_plus(y_ind,thetaE)
theta_m = theta_minus(y_ind,thetaE)
print(np.rad2deg(theta_p)*3600,np.rad2deg(theta_m)*3600)

ind_good = np.where(np.rad2deg(np.abs(theta_m))*3600>0.2)[0]



def scalar_product(hf, gf, psd, freqs):
    return 2.*simps( np.real((hf*np.conjugate(gf)+np.conjugate(hf)*gf))/psd, x=freqs)

ligo_psd = np.loadtxt('../data/AplusDesign.txt')
f_interp=ligo_psd[:,0]
ligo_interp = interp1d(f_interp,ligo_psd[:,1]**2)

virgo_psd = np.loadtxt('../data/V1_O5_strain.txt')
virgo_interp = interp1d(virgo_psd[:,0],virgo_psd[:,1]**2)

kagra_psd = np.loadtxt('../data/K1_O5_strain.txt')
kagra_interp = interp1d(kagra_psd[:,0],kagra_psd[:,1]**2)

det_list = {'H1':ligo_interp, 'L1':ligo_interp, 'V1':virgo_interp, 'K1':kagra_interp}

class GW_likelihood(bilby.Likelihood):
    def __init__(self,free_param,hf,waveform_generator,dets,f_low,injection_parameters):
        self.free_param = free_param
        self.hf = hf
        self.waveform_generator = waveform_generator
        self.dets = dets
        self.f_low = f_low
        self.injection_parameters = injection_parameters

        super().__init__(parameters={param: None for param in self.free_param})

    def log_likelihood(self):
        
        freq0 = self.waveform_generator.frequency_array

        wf_params_plus = dict(mass_1=self.parameters['m1'],mass_2=self.parameters['m2'],luminosity_distance=self.parameters['DL'],
                         theta_jn=self.parameters['iota'],a_1=0,a_2=0,tilt_1=0,tilt_2=0,phase=self.parameters['phic_plus'])
        h0_plus = self.waveform_generator.frequency_domain_strain(parameters=wf_params_plus)

        wf_params_minus = dict(mass_1=self.parameters['m1'],mass_2=self.parameters['m2'],luminosity_distance=self.parameters['DL'],
                         theta_jn=self.parameters['iota'],a_1=0,a_2=0,tilt_1=0,tilt_2=0,phase=self.parameters['phic_minus'])
        h0_minus = self.waveform_generator.frequency_domain_strain(parameters=wf_params_minus)

        tc_scale_plus = self.parameters['tc_plus']*1e-4
        tc_scale_minus = self.parameters['tc_minus']*1e-4
        ind = np.where(freq0>self.f_low)[0]

        loglike = 0
        if self.parameters['m1']<self.parameters['m2']:
            loglike = -np.inf
        else:
            for det in self.dets:
                ifo = bilby.gw.detector.InterferometerList([det])
                psd_interp = det_list[det]

                # Fp_plus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps']+tc_scale_plus, self.injection_parameters['psi'], 'plus')
                # Fx_plus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps']+tc_scale_plus, self.injection_parameters['psi'], 'cross')
                # Fp_minus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps_delay']+tc_scale_minus, self.injection_parameters['psi'], 'plus')
                # Fx_minus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps_delay']+tc_scale_minus, self.injection_parameters['psi'], 'cross')

                Fp_mod_plus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps'], self.injection_parameters['psi'], 'plus')
                Fx_mod_plus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps'], self.injection_parameters['psi'], 'cross')
                Fp_mod_minus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps_delay'], self.injection_parameters['psi'], 'plus')
                Fx_mod_minus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps_delay'], self.injection_parameters['psi'], 'cross')

                F_h0_plus = np.sqrt(np.abs(mu_plus(self.injection_parameters['y']))) * (Fp_mod_plus*h0_plus['plus']+Fx_mod_plus*h0_plus['cross']) * np.exp(-2.j*np.pi*freq0*tc_scale_plus)
                F_h0_minus = np.sqrt(np.abs(mu_minus(self.injection_parameters['y']))) * (Fp_mod_minus*h0_minus['plus']+Fx_mod_minus*h0_minus['cross']) * np.exp(-2.j*np.pi*freq0*tc_scale_minus)

                F_h_mod_plus = np.sqrt(np.abs(mu_plus(self.injection_parameters['y']))) * (Fp_mod_plus*self.hf['plus']+Fx_mod_plus*self.hf['cross'])
                F_h_mod_minus = np.sqrt(np.abs(mu_minus(self.injection_parameters['y']))) * (Fp_mod_minus*self.hf['plus']+Fx_mod_minus*self.hf['cross']) * np.exp(-1.j*(np.pi/2.))

                diff_plus = F_h0_plus[ind]-F_h_mod_plus[ind]
                diff_minus = F_h0_minus[ind]-F_h_mod_minus[ind]

                loglike -= 0.5*scalar_product(diff_plus,diff_plus,psd_interp(freq0[ind]),freq0[ind])
                loglike -= 0.5*scalar_product(diff_minus,diff_minus,psd_interp(freq0[ind]),freq0[ind])

        if(np.isnan(loglike)): loglike = -np.inf
        return loglike

f_low = 10
delta_f = 1./4
tc = 0
approx = 'IMRPhenomXPHM'
duration = 1/delta_f
sampling_frequency = 4096
free_param = ['m1','m2','DL','iota','tc_plus','tc_minus','phic_plus','phic_minus']

zarr = np.linspace(1.e-3,2,1000)
XdL = astropy_cosmo.luminosity_distance(zarr).value
priorD = XdL*XdL
YdL = priorD/sum(priorD)

for i in ind_good:
    t_delay = 4*np.pi*ML_ind[i]*MTSUN_SI*(1+zL_ind[i])*DeltaT(y_ind[i])

    dets = data['injections_parameters']['dets'][lensed_index][i]

    injection_parameters = dict(mass_1=data['injections_parameters']['m1d'][lensed_index][i], mass_2=data['injections_parameters']['m2d'][lensed_index][i],
                        ra=data['injections_parameters']['ras'][lensed_index][i], dec=data['injections_parameters']['decs'][lensed_index][i], psi=data['injections_parameters']['psis'][lensed_index][i], 
                        luminosity_distance=data['injections_parameters']['dls'][lensed_index][i], theta_jn=data['injections_parameters']['incs'][lensed_index][i], t_gps=data['injections_parameters']['geocent_time'][lensed_index][i],
                        t_gps_delay=data['injections_parameters']['geocent_time'][lensed_index][i]+t_delay, phase=data['injections_parameters']['phis'][lensed_index][i], y=y_ind[i], a_1=0, a_2=0, tilt_1=0, tilt_2=0)

    waveform_arguments = dict(waveform_approximant=approx,minimum_frequency=f_low)

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments)

    freq0 = waveform_generator.frequency_array
    h = waveform_generator.frequency_domain_strain(parameters=injection_parameters)

    priors = {}
    priors['m1'] = bilby.prior.Uniform(5.,500.,latex_label=r'$m_1$')
    priors['m2'] = bilby.prior.Uniform(5.,500.,latex_label=r'$m_2$')
    priors['DL'] = bilby.prior.Uniform(XdL[0], XdL[-1],latex_label=r'$d_L$')
    priors['iota'] = bilby.core.prior.Sine(minimum=0, maximum=np.pi,latex_label=r'$\iota$')
    priors['tc_plus'] = bilby.prior.Uniform(-100,100,latex_label=r'$t_{c,+}~[\times10^{-4}]$')
    priors['tc_minus'] = bilby.prior.Uniform(-100,100,latex_label=r'$t_{c,-}~[\times10^{-4}]$')
    priors['phic_plus'] = bilby.prior.Uniform(0,np.pi,latex_label=r'$\phi_{c,+}$')
    priors['phic_minus'] = bilby.prior.Uniform(0,np.pi,latex_label=r'$\phi_{c,-}$')
    # priors['ra'] = bilby.prior.Uniform(0,2*np.pi,latex_label=r'RA')
    # priors['dec'] = bilby.prior.Uniform(-np.pi/2,np.pi/2,latex_label=r'Dec')
    # priors['ML'] = bilby.prior.Uniform(1.,1000.,latex_label=r'$M_L$')
    # priors['y'] = bilby.prior.Uniform(0.001,1.,latex_label=r'$y$')

    likelihood=GW_likelihood(free_param,h,waveform_generator,dets,f_low,injection_parameters)

    result=bilby.run_sampler(likelihood, priors, sampler='nessai',nlive=700,naccept=60, check_point_delta_t=1800, print_method='interval-60', 
                            sample='acceptance-walk', npool=16,outdir='./sampling_sis_GO_seed_%d_ind_%d/'%(seed,lensed_index[i]),allow_multi_valued_likelihood = True)

    result.plot_corner()

