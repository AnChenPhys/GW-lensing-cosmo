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

CE_psd = np.loadtxt('../data/cosmic_explorer_strain.txt')
f_interp=CE_psd[:,0]
CE_interp = interp1d(f_interp,CE_psd[:,1]**2)

ET_psd = np.loadtxt('../data/18213_ET10kmcolumns.txt')
ET_interp = interp1d(ET_psd[:,0],ET_psd[:,1])

det_list = {'H1':CE_interp, 'L1':CE_interp, 'ET1':ET_interp, 'ET2':ET_interp, 'ET3':ET_interp}

def fisher_matrix(injection_parameters,waveform_generator,f_low,dets,delta=1e-3):

    Mc = injection_parameters['Mc']
    q = injection_parameters['q']
    m1 = ((1+q)/q**3)**(1./5)*Mc
    m2 = (q**2*(1+q))**(1./5)*Mc

    m1_dMc = ((1+q)/q**3)**(1./5)*(Mc+delta)
    m2_dMc = (q**2*(1+q))**(1./5)*(Mc+delta)
    m1_dq = ((1+(q+delta))/(q+delta)**3)**(1./5)*Mc
    m2_dq = ((q+delta)**2*(1+(q+delta)))**(1./5)*Mc

    wf_params = dict(mass_1=m1,mass_2=m2,luminosity_distance=injection_parameters['luminosity_distance'],
                        theta_jn=injection_parameters['theta_jn'],a_1=0,a_2=0,tilt_1=0,tilt_2=0,phase=injection_parameters['phase'])
    wf_params_dMc = dict(mass_1=m1_dMc,mass_2=m2_dMc,luminosity_distance=injection_parameters['luminosity_distance'],
                        theta_jn=injection_parameters['theta_jn'],a_1=0,a_2=0,tilt_1=0,tilt_2=0,phase=injection_parameters['phase'])
    wf_params_dq = dict(mass_1=m1_dq,mass_2=m2_dq,luminosity_distance=injection_parameters['luminosity_distance'],
                        theta_jn=injection_parameters['theta_jn'],a_1=0,a_2=0,tilt_1=0,tilt_2=0,phase=injection_parameters['phase'])
    # wf_params_ddL = dict(mass_1=m1,mass_2=m2,luminosity_distance=injection_parameters['luminosity_distance']+delta,
    #                     theta_jn=injection_parameters['theta_jn'],a_1=0,a_2=0,tilt_1=0,tilt_2=0,phase=injection_parameters['phase'])
    wf_params_dinc = dict(mass_1=m1,mass_2=m2,luminosity_distance=injection_parameters['luminosity_distance'],
                        theta_jn=injection_parameters['theta_jn']+delta,a_1=0,a_2=0,tilt_1=0,tilt_2=0,phase=injection_parameters['phase'])
    
    h0 = waveform_generator.frequency_domain_strain(parameters=wf_params)
    h0_dMc = waveform_generator.frequency_domain_strain(parameters=wf_params_dMc)
    h0_dq = waveform_generator.frequency_domain_strain(parameters=wf_params_dq)
    # h0_ddL = waveform_generator.frequency_domain_strain(parameters=wf_params_ddL)
    h0_dinc = waveform_generator.frequency_domain_strain(parameters=wf_params_dinc)

    freq0 = waveform_generator.frequency_array
    ind = np.where(freq0>f_low)[0]

    fisher_matrix = np.zeros((5,5))
    for det in dets:
        if det=='ET1' or det=='ET2' or det=='ET3':
            ifo = bilby.gw.detector.InterferometerList(['ET'])
            if det=='ET1':
                ifo_ind = 0
            elif det=='ET2':
                ifo_ind = 1
            elif det=='ET3':
                ifo_ind = 2
        else:
            ifo = bilby.gw.detector.InterferometerList([det])
            ifo_ind = 0
        psd_interp = det_list[det]

        Fp_mod_plus = ifo[ifo_ind].antenna_response(injection_parameters['ra'], injection_parameters['dec'], injection_parameters['t_gps'], injection_parameters['psi'], 'plus')
        Fx_mod_plus = ifo[ifo_ind].antenna_response(injection_parameters['ra'], injection_parameters['dec'], injection_parameters['t_gps'], injection_parameters['psi'], 'cross')
        Fp_mod_minus = ifo[ifo_ind].antenna_response(injection_parameters['ra'], injection_parameters['dec'], injection_parameters['t_gps_delay'], injection_parameters['psi'], 'plus')
        Fx_mod_minus = ifo[ifo_ind].antenna_response(injection_parameters['ra'], injection_parameters['dec'], injection_parameters['t_gps_delay'], injection_parameters['psi'], 'cross')

        F_h0_plus = np.sqrt(np.abs(mu_plus(injection_parameters['y']))) * (Fp_mod_plus*h0['plus']+Fx_mod_plus*h0['cross'])
        F_h0_minus = np.sqrt(np.abs(mu_minus(injection_parameters['y']))) * (Fp_mod_minus*h0['plus']+Fx_mod_minus*h0['cross']) * np.exp(-1.j*(np.pi/2.))
        F_h0_plus_dMc = np.sqrt(np.abs(mu_plus(injection_parameters['y']))) * (Fp_mod_plus*h0_dMc['plus']+Fx_mod_plus*h0_dMc['cross'])
        F_h0_minus_dMc = np.sqrt(np.abs(mu_minus(injection_parameters['y']))) * (Fp_mod_minus*h0_dMc['plus']+Fx_mod_minus*h0_dMc['cross']) * np.exp(-1.j*(np.pi/2.))
        F_h0_plus_dq = np.sqrt(np.abs(mu_plus(injection_parameters['y']))) * (Fp_mod_plus*h0_dq['plus']+Fx_mod_plus*h0_dq['cross'])
        F_h0_minus_dq = np.sqrt(np.abs(mu_minus(injection_parameters['y']))) * (Fp_mod_minus*h0_dq['plus']+Fx_mod_minus*h0_dq['cross']) * np.exp(-1.j*(np.pi/2.))
        # F_h0_plus_ddL = np.sqrt(np.abs(mu_plus(injection_parameters['y']))) * (Fp_mod_plus*h0_ddL['plus']+Fx_mod_plus*h0_ddL['cross'])
        # F_h0_minus_ddL = np.sqrt(np.abs(mu_minus(injection_parameters['y']))) * (Fp_mod_minus*h0_ddL['plus']+Fx_mod_minus*h0_ddL['cross']) * np.exp(-1.j*(np.pi/2.))
        F_h0_plus_dinc = np.sqrt(np.abs(mu_plus(injection_parameters['y']))) * (Fp_mod_plus*h0_dinc['plus']+Fx_mod_plus*h0_dinc['cross'])
        F_h0_minus_dinc = np.sqrt(np.abs(mu_minus(injection_parameters['y']))) * (Fp_mod_minus*h0_dinc['plus']+Fx_mod_minus*h0_dinc['cross']) * np.exp(-1.j*(np.pi/2.))
        F_h0_plus_dy = np.sqrt(np.abs(mu_plus(injection_parameters['y']+delta))) * (Fp_mod_plus*h0['plus']+Fx_mod_plus*h0['cross'])
        F_h0_minus_dy = np.sqrt(np.abs(mu_minus(injection_parameters['y']+delta))) * (Fp_mod_minus*h0['plus']+Fx_mod_minus*h0['cross']) * np.exp(-1.j*(np.pi/2.))

        diff_Mc_plus = (F_h0_plus_dMc-F_h0_plus)/delta
        diff_q_plus = (F_h0_plus_dq-F_h0_plus)/delta
        diff_dL_plus = -F_h0_plus/injection_parameters['luminosity_distance'] #(F_h0_plus_ddL-F_h0_plus)/delta
        diff_inc_plus = (F_h0_plus_dinc-F_h0_plus)/delta
        diff_y_plus = (F_h0_plus_dy-F_h0_plus)/delta
        diff_Mc_minus = (F_h0_minus_dMc-F_h0_minus)/delta
        diff_q_minus = (F_h0_minus_dq-F_h0_minus)/delta
        diff_dL_minus = -F_h0_minus/injection_parameters['luminosity_distance'] #(F_h0_minus_ddL-F_h0_minus)/delta
        diff_inc_minus = (F_h0_minus_dinc-F_h0_minus)/delta
        diff_y_minus = (F_h0_minus_dy-F_h0_minus)/delta

        diff_plus = [diff_Mc_plus[ind], diff_q_plus[ind], diff_dL_plus[ind], diff_inc_plus[ind], diff_y_plus[ind]]
        diff_minus = [diff_Mc_minus[ind], diff_q_minus[ind], diff_dL_minus[ind], diff_inc_minus[ind], diff_y_minus[ind]]

        for i in range(5):
            for j in range(5):
                fisher_matrix[i,j] += scalar_product(diff_plus[i],diff_plus[j],psd_interp(freq0[ind]),freq0[ind])
                fisher_matrix[i,j] += scalar_product(diff_minus[i],diff_minus[j],psd_interp(freq0[ind]),freq0[ind])

    return fisher_matrix


f_low = 10
delta_f = 1./4
tc = 0
approx = 'IMRPhenomXPHM'
duration = 1/delta_f
sampling_frequency = 4096
free_param = ['Mc','q','DL','iota','y']

ind_dict = np.load('event_ind_XG.npy',allow_pickle=True).item()

fr = open("/home/ansonchen/GW_lensing_dipole/XG_Tobs5_snr11/GW_injections_XG.p", "rb")
data = pickle.load(fr)
fr.close()

error_list = []
for s in range(2):
    error=[]
    np.random.seed(s)
    select_arr = []
    for k in range(len(ind_dict['lensed_index'][s])):
        select_arr.append(np.random.choice([0,1], size=1, p=[1-0.43,0.43])[0])
    visible_ind = np.where(ind_dict['lensed_index'][s] * np.array(select_arr)>0)[0]
    print(len(visible_ind))
    if np.size(visible_ind)==0:
        continue

    lensed_index = ind_dict['lensed_index'][s][visible_ind]
    ML_ind = ind_dict['ML_ind'][s][visible_ind]
    y_ind = ind_dict['y_ind'][s][visible_ind]
    zL_ind = ind_dict['zL_ind'][s][visible_ind]
    zS_ind = ind_dict['zS_ind'][s][visible_ind]

    for i in range(len(lensed_index)):
        t_delay = 4*np.pi*ML_ind[i]*MTSUN_SI*(1+zL_ind[i])*DeltaT(y_ind[i])

        dets = data['injections_parameters']['dets_pe'][lensed_index[i]]

        m1=data['injections_parameters']['m1d'][lensed_index[i]]
        m2=data['injections_parameters']['m2d'][lensed_index[i]]
        q = m2/m1
        Mc = m1 / ((1+q)/q**3)**(1./5)

        injection_parameters = dict(Mc=Mc, q=q,
                            ra=data['injections_parameters']['ras'][lensed_index[i]], dec=data['injections_parameters']['decs'][lensed_index[i]], psi=data['injections_parameters']['psis'][lensed_index[i]], 
                            luminosity_distance=data['injections_parameters']['dls'][lensed_index[i]], theta_jn=data['injections_parameters']['incs'][lensed_index[i]], t_gps=data['injections_parameters']['geocent_time'][lensed_index[i]],
                            t_gps_delay=data['injections_parameters']['geocent_time'][lensed_index[i]]+t_delay, phase=data['injections_parameters']['phis'][lensed_index[i]], y=y_ind[i], a_1=0, a_2=0, tilt_1=0, tilt_2=0)

        waveform_arguments = dict(waveform_approximant=approx,minimum_frequency=f_low)

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments)

        fisher = fisher_matrix(injection_parameters,waveform_generator,f_low,dets,delta=1e-3)

        cov_matrix = np.linalg.inv(fisher)
        error.append([np.sqrt(cov_matrix[0,0]),np.sqrt(cov_matrix[1,1]),np.sqrt(cov_matrix[2,2]),np.sqrt(cov_matrix[3,3]),np.sqrt(cov_matrix[4,4])])
    # error_list.append(error)

    np.savetxt('fisher_error_XG/fisher_error_XG_%d.txt'%s,error)
