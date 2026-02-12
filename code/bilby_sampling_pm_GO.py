import numpy as np
import matplotlib.pyplot as plt
import bilby
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftfreq
from lal import MTSUN_SI, PC_SI
from astropy.cosmology import FlatLambdaCDM
import matplotlib
plt.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True

# %config InlineBackend.figure_format = 'retina'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def scalar_product(hf, gf, psd, freqs):
    return 2.*simps( np.real((hf*np.conjugate(gf)+np.conjugate(hf)*gf))/psd, x=freqs)

class GW_likelihood(bilby.Likelihood):
    def __init__(self,free_param,F_h_plus_model,F_h_minus_model,waveform_generator,psd_interp_list,f_low):
        self.free_param = free_param
        self.F_h_plus_model = F_h_plus_model
        self.F_h_minus_model = F_h_minus_model
        self.waveform_generator = waveform_generator
        self.psd_interp_list = psd_interp_list
        self.f_low = f_low
        self.duration = duration
        self.sampling_frequency = sampling_frequency

        super().__init__(parameters={param: None for param in self.free_param})

    def log_likelihood(self):
        
        freq0 = self.waveform_generator.frequency_array

        wf_params = self.parameters
        wf_params['mass_1'] = ((1+self.parameters['q'])/self.parameters['q']**3)**(1./5)*self.parameters['chirp_mass']
        wf_params['mass_2'] = (self.parameters['q']**2*(1+self.parameters['q']))**(1./5)*self.parameters['chirp_mass']
        wf_params['a_1']=0
        wf_params['a_2']=0
        wf_params['tilt_1']=0
        wf_params['tilt_2']=0
        wf_params['phase']=0
        h0 = self.waveform_generator.frequency_domain_strain(parameters=wf_params)

        # ML = np.power(10,self.parameters['logML'])
        # w = 8*np.pi*ML*(1+zL)*MTSUN_SI*freq0
        # F_0 = F_geom_opt(w,self.parameters['y'])
        tc_scale = self.parameters['tc']*1e-4
        ind = np.where(freq0>self.f_low)[0]

        ifos = bilby.gw.detector.InterferometerList(['L1','H1','V1','K1'])

        loglike = 0
        for i in range(len(ifos)):
            Fp = ifos[i].antenna_response(self.parameters['ra'], self.parameters['dec'], t_gps, psi, 'plus')
            Fx = ifos[i].antenna_response(self.parameters['ra'], self.parameters['dec'], t_gps, psi, 'cross')

            F_h0_plus = np.sqrt(np.abs(mu_plus(self.parameters['y']))) * (Fp*h0['plus']+Fx*h0['cross']) * np.exp(-2.j*np.pi*freq0*tc_scale-1.j*self.parameters['phic']) #*np.conjugate(F_0)
            F_h0_minus = np.sqrt(np.abs(mu_minus(self.parameters['y']))) * (Fp*h0['plus']+Fx*h0['cross']) * np.exp(-2.j*np.pi*freq0*tc_scale-1.j*self.parameters['phic']-1.j*(np.pi/2.))
            
            diff_plus = F_h0_plus[ind]-self.F_h_plus_model[i][ind]
            diff_minus = F_h0_minus[ind]-self.F_h_minus_model[i][ind]

            psd_interp = self.psd_interp_list[i]
            loglike -= 0.5*scalar_product(diff_plus,diff_plus,psd_interp(freq0[ind]),freq0[ind])
            loglike -= 0.5*scalar_product(diff_minus,diff_minus,psd_interp(freq0[ind]),freq0[ind])

        if(np.isnan(loglike)): loglike = -np.inf
        return loglike


#Geometric optics - multiple images
def t_delay_geom_plus(y):
    return (y**2. + 2. - y*np.sqrt(y**2 +4.))/4.-np.log(np.abs(y+np.sqrt(y**2+4.))/2.)
def t_delay_geom_minus(y):
    return (y**2 + 2. + y*np.sqrt(y**2 +4.))/4.-np.log(np.abs(y-np.sqrt(y**2+4.))/2.)
def DeltaT(y):
    return t_delay_geom_minus(y)-t_delay_geom_plus(y)

def mu_plus(y):
    return 0.5 + (y**2. + 2.)/(2.*y*np.sqrt(y**2 + 4.))
def mu_minus(y):
    return 0.5 - (y**2. + 2.)/(2.*y*np.sqrt(y**2 + 4.))

def F_geom_opt(ws,y):
    Fplus = np.sqrt(np.abs(mu_plus(y)))*np.exp(1.0j*ws*t_delay_geom_plus(y))
    Fminus = np.sqrt(np.abs(mu_minus(y)))*np.exp(1.0j*ws*t_delay_geom_minus(y))*np.exp(-1.0j*(np.pi/2.)*np.sign(ws))
    return Fplus + Fminus


ligo_psd = np.loadtxt('../data/AplusDesign.txt')
f_interp=ligo_psd[:,0]
ligo_interp = interp1d(f_interp,ligo_psd[:,1]**2)

virgo_psd = np.loadtxt('../data/V1_O5_strain.txt')
virgo_interp = interp1d(virgo_psd[:,0],virgo_psd[:,1]**2)

kagra_psd = np.loadtxt('../data/K1_O5_strain.txt')
kagra_interp = interp1d(kagra_psd[:,0],kagra_psd[:,1]**2)

psd_interp_list = [ligo_interp,ligo_interp,virgo_interp,kagra_interp]

yh = 1
zL = 0.05
zS = 0.1
H0 = 67.7
Om0 = 0.308
inc = np.pi/3
ML_GO = 1e7

m1 = 36*(1+zS)
m2 = 29*(1+zS)
Mc = (m1*m2)**(3/5)/(m1 + m2)**(1/5)
q = m2/m1

ra = np.deg2rad(26)
dec = np.deg2rad(48)
t_gps=30000
psi=0

f_low = 10
delta_f = 1./4
tc = 0
approx = 'IMRPhenomXPHM'

cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)
dls = cosmo.luminosity_distance(zS).value

duration = 1/delta_f
sampling_frequency = 4096

free_param = ['chirp_mass','q','luminosity_distance','theta_jn','tc','phic','ra','dec','y']
parameters = dict(mass_1=m1, mass_2=m2, luminosity_distance=dls, theta_jn=inc, phase=0, a_1=0, a_2=0, tilt_1=0, tilt_2=0)#, logML=np.log10(ML_GO), y=yh)

waveform_arguments = dict(waveform_approximant=approx,minimum_frequency=f_low)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

freq0 = waveform_generator.frequency_array

h = waveform_generator.frequency_domain_strain(parameters=parameters)

# w = 8*np.pi*ML_GO*(1+zL)*MTSUN_SI*freq0

# F_0_GO = F_geom_opt(w,yh)
# F_h_0_GO = (h['plus']+h['cross'])*np.conjugate(F_0_GO)

F_h_GO_plus = []
F_h_GO_minus = []

ifos = bilby.gw.detector.InterferometerList(['L1','H1','V1','K1'])
for i in range(len(ifos)):
    Fp = ifos[i].antenna_response(ra, dec, t_gps, psi, 'plus')
    Fx = ifos[i].antenna_response(ra, dec, t_gps, psi, 'cross')

    F_h_GO_plus.append(np.sqrt(np.abs(mu_plus(yh)))*(Fp*h['plus']+Fx*h['cross']))
    F_h_GO_minus.append(np.sqrt(np.abs(mu_minus(yh)))*(Fp*h['plus']+Fx*h['cross'])*np.exp(-1.0j*(np.pi/2.)))

zarr = np.linspace(1.e-3,1,1000)
XdL = cosmo.luminosity_distance(zarr).value
priorD = XdL*XdL
YdL = priorD/sum(priorD)

priors = {}
priors['chirp_mass'] = bilby.prior.Uniform(5.,100.,latex_label=r'${\cal M}_c$')
priors['q'] = bilby.prior.Uniform(0.01,1.,latex_label=r'$q$')
priors['luminosity_distance'] = bilby.prior.Uniform(XdL[0], XdL[-1],latex_label=r'$d_L^s$')
# priors['luminosity_distance'] = bilby.core.prior.Interped(xx=XdL,yy=YdL,minimum=XdL[0], maximum=XdL[-1],latex_label=r'$d_L^s$')
priors['theta_jn'] = bilby.core.prior.Sine(minimum=0, maximum=np.pi/2,latex_label=r'$\iota$')
priors['tc'] = bilby.prior.Uniform(-100,100,latex_label=r'$t_c~[\times10^{-4}]$')
priors['phic'] = bilby.prior.Uniform(-np.pi,np.pi,latex_label=r'$\phi_c$')
priors['ra'] = bilby.prior.Uniform(0,2*np.pi,latex_label=r'RA')
priors['dec'] = bilby.prior.Uniform(-np.pi/2,np.pi/2,latex_label=r'Dec')
# priors['logML'] = bilby.prior.Uniform(5.,9.,latex_label=r'$\log_{10} M_L$')
priors['y'] = bilby.prior.Uniform(0.01,5.,latex_label=r'$y$')

likelihood=GW_likelihood(free_param,F_h_GO_plus,F_h_GO_minus,waveform_generator,psd_interp_list,f_low)

result=bilby.run_sampler(likelihood, priors, sampler='nessai',nlive=700,naccept=60, check_point_delta_t=1800, print_method='interval-60', 
                         sample='acceptance-walk', npool=16,outdir='./results_pm_GO/',allow_multi_valued_likelihood = True)

result.plot_corner()
