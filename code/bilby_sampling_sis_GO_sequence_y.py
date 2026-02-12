# import icarogw
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps
import scipy.constants as constants
from lal import MTSUN_SI, PC_SI
from astropy.cosmology import FlatLambdaCDM
import bilby
import pickle
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--n', type='int', dest='n')
(options, args) = parser.parse_args()
n = options.n

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
def scalar_product(hf, gf, psd, freqs):
    return 2.*simps( np.real((hf*np.conjugate(gf)+np.conjugate(hf)*gf))/psd, x=freqs)


Om0 = 0.308
H0 = 67.7
astropy_cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)

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

        Mc = self.parameters['Mc']
        q = self.parameters['q']
        m1 = ((1+q)/q**3)**(1./5)*Mc
        m2 = (q**2*(1+q))**(1./5)*Mc

        wf_params = dict(mass_1=m1,mass_2=m2,luminosity_distance=self.parameters['DL'],
                         theta_jn=self.parameters['iota'],a_1=0,a_2=0,tilt_1=0,tilt_2=0,phase=self.injection_parameters['phase'])
        h0 = self.waveform_generator.frequency_domain_strain(parameters=wf_params)

        freq0 = self.waveform_generator.frequency_array
        ind = np.where(freq0>self.f_low)[0]

        loglike = 0
        for det in self.dets:
            ifo = bilby.gw.detector.InterferometerList([det])
            psd_interp = det_list[det]

            Fp_mod_plus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps'], self.injection_parameters['psi'], 'plus')
            Fx_mod_plus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps'], self.injection_parameters['psi'], 'cross')
            Fp_mod_minus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps_delay'], self.injection_parameters['psi'], 'plus')
            Fx_mod_minus = ifo[0].antenna_response(self.injection_parameters['ra'], self.injection_parameters['dec'], self.injection_parameters['t_gps_delay'], self.injection_parameters['psi'], 'cross')

            F_h0_plus = np.sqrt(np.abs(mu_plus(self.parameters['y']))) * (Fp_mod_plus*h0['plus']+Fx_mod_plus*h0['cross'])
            F_h0_minus = np.sqrt(np.abs(mu_minus(self.parameters['y']))) * (Fp_mod_minus*h0['plus']+Fx_mod_minus*h0['cross']) * np.exp(-1.j*(np.pi/2.))

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
free_param = ['Mc','q','DL','iota','y']

zarr = np.linspace(1.e-3,5,1000)
XdL = astropy_cosmo.luminosity_distance(zarr).value
priorD = XdL*XdL
YdL = priorD/sum(priorD)

yarr = np.linspace(1e-3,1,1000)
priory = yarr/sum(yarr)

ind_dict = np.load('event_ind.npy',allow_pickle=True).item()

fr = open("../O5_Tobs5_snr11/GW_injections_O5.p", "rb")
data = pickle.load(fr)
fr.close()

for s in range(n*200,(n+1)*200):
    if s >= len(ind_dict['seed']):
        break

    seed = ind_dict['seed'][s]
    np.random.seed(seed)
    select_arr = []
    for k in range(len(ind_dict['lensed_index'][s])):
        select_arr.append(np.random.choice([0,1], size=1, p=[1-0.43,0.43])[0])
    visible_ind = np.where(ind_dict['lensed_index'][s] * np.array(select_arr)>0)[0]
    print(visible_ind)
    if np.size(visible_ind)==0:
        continue

    lensed_index = ind_dict['lensed_index'][s][visible_ind]
    ML_ind = ind_dict['ML_ind'][s][visible_ind]
    y_ind = ind_dict['y_ind'][s][visible_ind]
    zL_ind = ind_dict['zL_ind'][s][visible_ind]
    zS_ind = ind_dict['zS_ind'][s][visible_ind]

    for i in range(len(lensed_index)):
        t_delay = 4*np.pi*ML_ind[i]*MTSUN_SI*(1+zL_ind[i])*DeltaT(y_ind[i])

        dets = data['injections_parameters']['dets'][lensed_index[i]]

        injection_parameters = dict(mass_1=data['injections_parameters']['m1d'][lensed_index[i]], mass_2=data['injections_parameters']['m2d'][lensed_index[i]],
                            ra=data['injections_parameters']['ras'][lensed_index[i]], dec=data['injections_parameters']['decs'][lensed_index[i]], psi=data['injections_parameters']['psis'][lensed_index[i]], 
                            luminosity_distance=data['injections_parameters']['dls'][lensed_index[i]], theta_jn=data['injections_parameters']['incs'][lensed_index[i]], t_gps=data['injections_parameters']['geocent_time'][lensed_index[i]],
                            t_gps_delay=data['injections_parameters']['geocent_time'][lensed_index[i]]+t_delay, phase=data['injections_parameters']['phis'][lensed_index[i]], y=y_ind[i], a_1=0, a_2=0, tilt_1=0, tilt_2=0)

        waveform_arguments = dict(waveform_approximant=approx,minimum_frequency=f_low)

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments)

        freq0 = waveform_generator.frequency_array
        h = waveform_generator.frequency_domain_strain(parameters=injection_parameters)

        priors = {}
        # priors['m1'] = bilby.prior.Uniform(5.,500.,latex_label=r'$m_1$')
        # priors['m2'] = bilby.prior.Uniform(5.,500.,latex_label=r'$m_2$')
        # priors['DL'] = bilby.prior.Uniform(XdL[0], XdL[-1],latex_label=r'$d_L$')
        priors['Mc'] = bilby.prior.Uniform(5.,500.,latex_label=r'$M_c$')
        priors['q'] = bilby.prior.Uniform(0.05,1.,latex_label=r'$q$')
        priors['DL'] = bilby.prior.Interped(xx=XdL,yy=YdL,minimum=XdL[0], maximum=XdL[-1],latex_label=r'$d_L$')
        priors['iota'] = bilby.core.prior.Sine(minimum=0, maximum=np.pi,latex_label=r'$\iota$')
        # priors['tc_plus'] = bilby.prior.Uniform(-100,100,latex_label=r'$t_{c,+}~[\times10^{-4}]$')
        # priors['tc_minus'] = bilby.prior.Uniform(-100,100,latex_label=r'$t_{c,-}~[\times10^{-4}]$')
        # priors['phic_plus'] = bilby.prior.Uniform(0,np.pi,latex_label=r'$\phi_{c,+}$')
        # priors['phic_minus'] = bilby.prior.Uniform(0,np.pi,latex_label=r'$\phi_{c,-}$')
        # priors['ra'] = bilby.prior.Uniform(0,2*np.pi,latex_label=r'RA')
        # priors['dec'] = bilby.prior.Uniform(-np.pi/2,np.pi/2,latex_label=r'Dec')
        # priors['ML'] = bilby.prior.Uniform(1.,1000.,latex_label=r'$M_L$')
        priors['y'] = bilby.prior.Interped(xx=yarr,yy=priory, minimum=0.001, maximum=1., latex_label=r'$y$')

        likelihood=GW_likelihood(free_param,h,waveform_generator,dets,f_low,injection_parameters)

        result=bilby.run_sampler(likelihood, priors, sampler='nessai',nlive=750,naccept=60, check_point_delta_t=1800, print_method='interval-60', 
                                sample='acceptance-walk', npool=16, outdir='../samples_y/sampling_sis_GO_seed_%d_ind_%d/'%(seed,lensed_index[i]),allow_multi_valued_likelihood = True)

        result.plot_corner()

