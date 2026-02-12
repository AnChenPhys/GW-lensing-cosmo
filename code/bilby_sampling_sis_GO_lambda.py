import numpy as np
import matplotlib.pyplot as plt
import bilby
from scipy.integrate import simps
from scipy.interpolate import interp1d
from lal import MTSUN_SI, PC_SI
from astropy.cosmology import FlatLambdaCDM
import matplotlib
plt.rcParams.update({'font.size': 15})
# matplotlib.rcParams['text.usetex'] = True

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

def t_delay_geom_plus(y):
    return -y-0.5
def t_delay_geom_minus(y):
    return y-0.5
def DeltaT(y):
    return t_delay_geom_minus(y)-t_delay_geom_plus(y)

def mu_plus(y):
    return 1+1/y
def mu_minus(y):
    return -1+1/y

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

        wf_params = dict(mass_1=self.parameters['m1'],mass_2=self.parameters['m2'],luminosity_distance=self.parameters['DL'],
                         theta_jn=self.parameters['iota'],a_1=0,a_2=0,tilt_1=0,tilt_2=0,phase=self.parameters['phic'])

        h0 = self.waveform_generator.frequency_domain_strain(parameters=wf_params)

        tc_scale = self.parameters['tc']*1e-4
        ind = np.where(freq0>self.f_low)[0]

        loglike = 0
        if self.parameters['m1']<self.parameters['m2']:
            loglike = -np.inf
        else:
            for det in self.dets:
                ifo = bilby.gw.detector.InterferometerList([det])
                psd_interp = det_list[det]

                Fp_plus = ifo[0].antenna_response(self.parameters['ra'], self.parameters['dec'], self.injection_parameters['t_gps'], self.injection_parameters['psi'], 'plus')
                Fx_plus = ifo[0].antenna_response(self.parameters['ra'], self.parameters['dec'], self.injection_parameters['t_gps'], self.injection_parameters['psi'], 'cross')
                Fp_minus = ifo[0].antenna_response(self.parameters['ra'], self.parameters['dec'], self.injection_parameters['t_gps_delay'], self.injection_parameters['psi'], 'plus')
                Fx_minus = ifo[0].antenna_response(self.parameters['ra'], self.parameters['dec'], self.injection_parameters['t_gps_delay'], self.injection_parameters['psi'], 'cross')

                Fp_mod_plus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps'], self.injection_parameters['psi'], 'plus')
                Fx_mod_plus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps'], self.injection_parameters['psi'], 'cross')
                Fp_mod_minus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps_delay'], self.injection_parameters['psi'], 'plus')
                Fx_mod_minus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps_delay'], self.injection_parameters['psi'], 'cross')

                F_h0_plus = np.sqrt(np.abs(mu_plus(self.parameters['y']))) / self.parameters['lambda'] * (Fp_plus*h0['plus']+Fx_plus*h0['cross']) * np.exp(-2.j*np.pi*freq0*tc_scale)
                F_h0_minus = np.sqrt(np.abs(mu_minus(self.parameters['y']))) / self.parameters['lambda'] * (Fp_minus*h0['plus']+Fx_minus*h0['cross']) * np.exp(-2.j*np.pi*freq0*tc_scale-1.j*(np.pi/2.))

                F_h_mod_plus = np.sqrt(np.abs(mu_plus(self.injection_parameters['y']))) * (Fp_mod_plus*h0['plus']+Fx_mod_plus*h0['cross'])
                F_h_mod_minus = np.sqrt(np.abs(mu_minus(self.injection_parameters['y']))) * (Fp_mod_minus*self.hf['plus']+Fx_mod_minus*self.hf['cross']) * np.exp(-1.j*(np.pi/2.))

                diff_plus = F_h0_plus[ind]-F_h_mod_plus[ind]
                diff_minus = F_h0_minus[ind]-F_h_mod_minus[ind]

                loglike -= 0.5*scalar_product(diff_plus,diff_plus,psd_interp(freq0[ind]),freq0[ind])
                loglike -= 0.5*scalar_product(diff_minus,diff_minus,psd_interp(freq0[ind]),freq0[ind])

        if(np.isnan(loglike)): loglike = -np.inf
        return loglike

yh = 0.5
zL = 0.1
zS = 0.2
H0 = 67.7
Om0 = 0.308
inc = np.pi/3
ML = 1e11

m1 = 36*(1+zS)
m2 = 29*(1+zS)
Mc = (m1*m2)**(3/5)/(m1 + m2)**(1/5)
q = m2/m1

f_low = 10
delta_f = 1./16
tc = 0
approx = 'IMRPhenomXPHM'

ra = np.deg2rad(26)
dec = np.deg2rad(48)
t_gps=30000
psi=0

t_delay = 4*np.pi*ML*MTSUN_SI*(1+zL)*DeltaT(yh)

cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)
dls = cosmo.luminosity_distance(zS).value

duration = 1/delta_f
sampling_frequency = 4096

dets = ['L1','H1','V1','K1']

free_param = ['chirp_mass','q','luminosity_distance','theta_jn','tc','phic','ra','dec','y','lambda']
injection_parameters = dict(mass_1=m1, mass_2=m2, luminosity_distance=dls, theta_jn=inc, phase=0, a_1=0, a_2=0, tilt_1=0, tilt_2=0,
                            ra=ra, dec=dec, t_gps=t_gps, t_gps_delay=t_gps+t_delay, psi=psi, y=yh)#, logML=np.log10(ML_GO), y=yh)

waveform_arguments = dict(waveform_approximant=approx,minimum_frequency=f_low)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

freq0 = waveform_generator.frequency_array

h = waveform_generator.frequency_domain_strain(parameters=injection_parameters)

zarr = np.linspace(1.e-3,2,1000)
cosmo = FlatLambdaCDM(H0=67.7,Om0=0.308)
XdL = cosmo.luminosity_distance(zarr).value
priorD = XdL*XdL
YdL = priorD/sum(priorD)

priors = {}
priors['m1'] = bilby.prior.Uniform(5.,500.,latex_label=r'$m_1$')
priors['m2'] = bilby.prior.Uniform(5.,500.,latex_label=r'$m_2$')
priors['DL'] = bilby.prior.Uniform(XdL[0], XdL[-1],latex_label=r'$d_L$')
priors['iota'] = bilby.core.prior.Sine(minimum=0, maximum=np.pi/2,latex_label=r'$\iota$')
priors['tc'] = bilby.prior.Uniform(-100,100,latex_label=r'$t_c~[\times10^{-4}]$')
priors['phic'] = bilby.prior.Uniform(-np.pi,np.pi,latex_label=r'$\phi_c$')
priors['ra'] = bilby.prior.Uniform(0,2*np.pi,latex_label=r'RA')
priors['dec'] = bilby.prior.Uniform(-np.pi/2,np.pi/2,latex_label=r'Dec')
# priors['ML'] = bilby.prior.Uniform(1.,1000.,latex_label=r'$M_L$')
priors['y'] = bilby.prior.Uniform(0.001,1.,latex_label=r'$y$')
priors['lambda'] = bilby.prior.Uniform(0.001,10.,latex_label=r'$\lambda$')

likelihood=GW_likelihood(free_param,h,waveform_generator,dets,f_low,injection_parameters)

result=bilby.run_sampler(likelihood, priors, sampler='nessai',nlive=700,naccept=60, check_point_delta_t=1800, print_method='interval-60', 
                         sample='acceptance-walk', npool=32,outdir='./results_sis_GO_lambda/',allow_multi_valued_likelihood = True)

result.plot_corner()

