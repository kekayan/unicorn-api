# import libraries
import os
import sys
import re

# data science imports
import pandas as pd
import numpy as np

# signal processing imports
import scipy.signal
from scipy import signal, sparse
from six.moves import xrange
from scipy.sparse.linalg import spsolve
from scipy.stats import kurtosis
from scipy.stats import skew
import scipy.fftpack
from scipy.signal import butter, lfilter, detrend

import scipy.signal as sig
from scipy.signal import detrend

import neurokit2 as nk


# bandpower features
bandpower_features_Headers = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
# Select the EEG electrodes of interest
eeg_data = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']

SAMPLING_RATE = 250 # Hz


eeg_bandpower_features = ['FZ_Delta', 'FZ_Theta', 'FZ_Alpha', 'FZ_Beta', 'FZ_Gamma', 
                          'C3_Delta', 'C3_Theta', 'C3_Alpha', 'C3_Beta', 'C3_Gamma',
                          'CZ_Delta', 'CZ_Theta', 'CZ_Alpha', 'CZ_Beta', 'CZ_Gamma',
                          'C4_Delta', 'C4_Theta', 'C4_Alpha', 'C4_Beta', 'C4_Gamma',
                          'PZ_Delta', 'PZ_Theta', 'PZ_Alpha', 'PZ_Beta', 'PZ_Gamma',
                          'PO7_Delta', 'PO7_Theta', 'PO7_Alpha', 'PO7_Beta', 'PO7_Gamma',
                          'OZ_Delta', 'OZ_Theta', 'OZ_Alpha', 'OZ_Beta', 'OZ_Gamma',
                          'PO8_Delta', 'PO8_Theta', 'PO8_Alpha', 'PO8_Beta', 'PO8_Gamma']


def _signal_filter_sanitize(lowcut=None, highcut=None, sampling_rate=250, normalize=False):
    # Sanity checks
    # if isinstance(highcut, int):
    # if sampling_rate <= 2 * highcut:
    #     warn(
    #         "The sampling rate is too low. Sampling rate"
    #         " must exceed the Nyquist rate to avoid aliasing problem."
    #         f" In this analysis, the sampling rate has to be higher than {2 * highcut} Hz",
    #         category=NeuroKitWarning,
    #     )

    # Replace 0 by none
    if lowcut is not None and lowcut == 0:
        lowcut = None
    if highcut is not None and highcut == 0:
        highcut = None

    # Format
    if lowcut is not None and highcut is not None:
        if lowcut > highcut:
            filter_type = "bandstop"
        else:
            filter_type = "bandpass"
        freqs = [lowcut, highcut]
    elif lowcut is not None:
        freqs = [lowcut]
        filter_type = "highpass"
    elif highcut is not None:
        freqs = [highcut]
        filter_type = "lowpass"

    # Normalize frequency to Nyquist Frequency (Fs/2).
    # However, no need to normalize if `fs` argument is provided to the scipy filter
    if normalize is True:
        freqs = np.array(freqs) / (sampling_rate / 2)

    return freqs, filter_type


'''
Butterworth filter to remove noise specially artifact noise
'''


def butterworth_filter(data, sampling_rate=250, lowcut=None, highcut=None, order=2):
    """Filter a signal using IIR Butterworth SOS method."""
    freqs, filter_type = _signal_filter_sanitize(lowcut=lowcut, highcut=highcut, sampling_rate=sampling_rate)

    sos = scipy.signal.butter(order, freqs, btype=filter_type, output="sos", fs=sampling_rate)
    filtered = scipy.signal.sosfiltfilt(sos, data)
    return filtered


def demean(x, axis=0, bp=0):
    """
    Demean array along one axis.

    :param x: array to demean. Remains unchanged.
    :type x: array
    :param axis: the axis along which to demean. Default is 0, not -1 as in scipy!
    :type axis: int
    :param bp: A sequence of break points. If given, an individual fit is
               performed for each part of `x` between two break points.
               Break points are specified as indices into `x`.
    :type bp: array_like of ints, optional

    :returns: Demeaned array.
    """
    return detrend(x, axis, "constant", bp)


def eeg_power_sanitize(frequency_band=None):
    if frequency_band is None:
        frequency_band = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        
    band_names = frequency_band.copy()

    for i, f in enumerate(frequency_band):
        if isinstance(f, str):
            f_name = f.lower()
            if f_name == "gamma":
                frequency_band[i] = (30, 45)
            elif f_name == "beta":
                frequency_band[i] = (13, 30)
            elif f_name == "beta1":
                frequency_band[i] = (13, 16)
            elif f_name == "beta2":
                frequency_band[i] = (16, 20)
            elif f_name == "beta3":
                frequency_band[i] = (20, 30)
            elif f_name == "smr":
                frequency_band[i] = (13, 15)
            elif f_name == "alpha":
                frequency_band[i] = (8, 13)
            elif f_name == "mu":
                frequency_band[i] = (9, 11)
            elif f_name == "theta":
                frequency_band[i] = (4, 8)
            elif f_name == "delta":
                frequency_band[i] = (1, 4)
            else:
                raise ValueError(f"Unknown frequency band: '{f_name}'")
        elif isinstance(f, tuple):
            band_names[i] = f"Hz_{f[0]}_{f[1]}"
        else:
            raise ValueError("'frequency_band' must be a list of tuples (or strings).")
    return frequency_band, band_names



def set_channelFreqFeatures(eeg_freq):
    s = pd.Series(index=eeg_bandpower_features)


    for i in range(len(eeg_freq)):
        data = eeg_freq[i]
        # print(len(data))
        if i == 0:
            # print("Setting FZ Features")
            s['FZ_Delta'] = data[4]
            s['FZ_Theta'] = data[3]
            s['FZ_Alpha'] = data[2]
            s['FZ_Beta'] = data[1]
            s['FZ_Gamma'] = data[0]
        
        elif i == 1:
            # print("Setting C3 Features")
            s['C3_Delta'] = data[4]
            s['C3_Theta'] = data[3]
            s['C3_Alpha'] = data[2]
            s['C3_Beta'] = data[1]
            s['C3_Gamma'] = data[0]
            
        elif i == 2:
            # print("Setting CZ Features")
            s['CZ_Delta'] = data[4]
            s['CZ_Theta'] = data[3]
            s['CZ_Alpha'] = data[2]
            s['CZ_Beta'] = data[1]
            s['CZ_Gamma'] = data[0]
        
        elif i == 3:
            # print("Setting C4 Features")
            s['C4_Delta'] = data[4]
            s['C4_Theta'] = data[3]
            s['C4_Alpha'] = data[2]
            s['C4_Beta'] = data[1]
            s['C4_Gamma'] = data[0]    
            
        elif i == 4:
            # print("Setting PZ Features")
            s['PZ_Delta'] = data[4]
            s['PZ_Theta'] = data[3]
            s['PZ_Alpha'] = data[2]
            s['PZ_Beta'] = data[1]
            s['PZ_Gamma'] = data[0]
        elif i == 5:
            # print("Setting PO7 Features")
            s['PO7_Delta'] = data[4]
            s['PO7_Theta'] = data[3]
            s['PO7_Alpha'] = data[2]
            s['PO7_Beta'] = data[1]
            s['PO7_Gamma'] = data[0]
        elif i == 6:
            # print("Setting OZ Features")
            s['OZ_Delta'] = data[4]
            s['OZ_Theta'] = data[3]
            s['OZ_Alpha'] = data[2]
            s['OZ_Beta'] = data[1]
            s['OZ_Gamma'] = data[0]
        elif i == 7:
            # print("Setting PO8 Features")
            s['PO8_Delta'] = data[4]
            s['PO8_Theta'] = data[3]
            s['PO8_Alpha'] = data[2]
            s['PO8_Beta'] = data[1]
            s['PO8_Gamma'] = data[0]


    return s
    

# revised by zhuang
# 0927
def get_avg_power(data):
    
    # alpha = ((data['FZ_Alpha'] + data['C3_Alpha'] + data['CZ_Alpha'] + data['C4_Alpha'] + data['PZ_Alpha'] + data['PO7_Alpha'] + data['OZ_Alpha'] + data['PO8_Alpha'])/8)
    # beta = ((data['FZ_Beta'] + data['C3_Beta'] + data['CZ_Beta'] + data['C4_Beta'] + data['PZ_Beta'] + data['PO7_Beta'] + data['OZ_Beta'] + data['PO8_Beta'])/8)
    # theta = ((data['FZ_Theta'] + data['C3_Theta'] + data['CZ_Theta'] + data['C4_Theta'] + data['PZ_Theta'] + data['PO7_Theta'] + data['OZ_Theta'] + data['PO8_Theta'])/8)
    # gamma = ((data['FZ_Gamma'] + data['C3_Gamma'] + data['CZ_Gamma'] + data['C4_Gamma'] + data['PZ_Gamma'] + data['PO7_Gamma'] + data['OZ_Gamma'] + data['PO8_Gamma'])/8)
    # delta = ((data['FZ_Delta'] + data['C3_Delta'] + data['CZ_Delta'] + data['C4_Delta'] + data['PZ_Delta'] + data['PO7_Delta'] + data['OZ_Delta'] + data['PO8_Delta'])/8)
    # fz_alpha = data['FZ_Alpha']
    # fz_beta = data['FZ_Beta']
    # fz_theta = data['FZ_Theta']
    # fz_gamma = data['FZ_Gamma']
    # fz_delta = data['FZ_Delta']
    # all_band_power_array =  [fz_alpha, fz_beta,  fz_delta, fz_gamma , fz_theta]                #[alpha, beta, theta, gamma, delta] 
    
    #zhuang: Unicorn electrode
    # Alpha = data['C3_Alpha'] + data['CZ_Alpha'] + data['C4_Alpha'] + data['PZ_Alpha']
    # Beta =  data['C3_Beta'] +  data['CZ_Beta'] +  data['C4_Beta'] +  data['PZ_Beta']
    # Theta = data['C3_Theta'] + data['CZ_Theta'] + data['C4_Theta'] + data['PZ_Theta']
    # Gamma = data['C3_Gamma'] + data['CZ_Gamma'] + data['C4_Gamma'] + data['PZ_Gamma']
    # Delta = data['C3_Delta'] + data['CZ_Delta'] + data['C4_Delta'] + data['PZ_Delta']
    
    #zhuang: Unicorn Gtech
    # OZ-Pz
    # CZ-P3
    # PO7-P4
    # PO8-Cz
    Alpha = data['OZ_Alpha'] + data['CZ_Alpha'] + data['PO7_Alpha'] + data['PO8_Alpha']
    Beta =  data['OZ_Beta'] +  data['CZ_Beta'] +  data['PO7_Beta'] +  data['PO8_Beta']
    Theta = data['OZ_Theta'] + data['CZ_Theta'] + data['PO7_Theta'] + data['PO8_Theta']
    Gamma = data['OZ_Gamma'] + data['CZ_Gamma'] + data['PO7_Gamma'] + data['PO8_Gamma']
    Delta = data['OZ_Delta'] + data['CZ_Delta'] + data['PO7_Delta'] + data['PO8_Delta']

    all_band_power_array =  [Alpha, Beta,  Delta, Gamma , Theta]                #[alpha, beta, theta, gamma, delta]    

    return all_band_power_array


# revised by zhuang
# 0927
def engagemnet_factor(all_band_power_array):

    # eng_1 = all_band_power_array[4]/all_band_power_array[0]
    # eng_2 = all_band_power_array[1] - eng_1
    # engagemnet_array = [eng_1, eng_2]

    eng_1 = all_band_power_array[1]/(all_band_power_array[0] + all_band_power_array[4]) # beta/(alpha+theta)
    eng_2 = all_band_power_array[1]/(all_band_power_array[0] + all_band_power_array[4]) # beta/alpha
    engagemnet_array = [eng_1, eng_2]

    return engagemnet_array 




def get_power(data, sampling_freq=250):
   
    
    #freq_band, band_names = eeg_power_sanitize(frequency_band=["Gamma", "Beta", "Alpha", "Theta", "Delta"])
    power_data = []
    freq_band = [(30, 45), (13, 30), (8, 13), (4, 8), (1, 4)]
    for i in range(len(data)):
        rez = nk.signal_power(data[i],frequency_band=freq_band, sampling_rate=250, method='welch')
        temp = rez.values
        power_data.append(temp[0])
   
 

    data_array = np.asarray(power_data)
    out = set_channelFreqFeatures(data_array)
    all_band_power_array = get_avg_power(out)
    return all_band_power_array
    # engagemnet_factor_array = engagemnet_factor(all_band_power_array)
    
    

    # return engagemnet_factor_array


def eeg_features_extract(data, sampling_freq=250):
    filtered_data = []

    # print(bads, info)
    data = np.asarray(data)


    for column in data.T:

      
        raw = column
        raw_prep = demean(raw)
        prep_data = butterworth_filter(raw_prep, sampling_rate=sampling_freq, lowcut=1.0, highcut=50)
        
        filtered_data.append(prep_data)
        
    filtered_data = np.asarray(filtered_data)

    all_band_power_array = get_power(filtered_data, sampling_freq)

    final_series = all_band_power_array


    return final_series