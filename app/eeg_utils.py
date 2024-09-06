import numpy as np
from scipy.signal import butter, lfilter, iirnotch, filtfilt,welch

# ----------- funcs ---------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def apply_notch_filter(data, freq, fs, quality=30):
    w0 = freq/(0.5*fs)
    b, a = iirnotch(w0, quality)
    y = filtfilt(b, a, data)
    return y
# ---------------------------------


# # df = pd.read_csv('unicorn_data.csv', header=None)
# df = pd.read_csv('data_new_unicorn.txt', sep=' ', header=None)

# # df.fillna(df.mean(), inplace=True)
# df = df.apply(pd.to_numeric, errors='coerce').fillna(method='ffill')

# print("Shape of the DataFrame:", df.shape)

# accelerometer = np.array(df.iloc[:, 8:11])  # Accelerometer data (x, y, z)
# gyroscope = np.array(df.iloc[:, 11:13])  # Gyroscope data (x, y, z)
# battery_level = np.array(df.iloc[:, 14])  # Battery level
# counter = np.array(df.iloc[:,15])  # Counter
# validation_indicator = np.array(df.iloc[:, 16])  # Validation indicator

def extract_eeg_bands(data):

    eeg_channels = data
    # CAR
    mean_across_channels = np.mean(eeg_channels, axis=1, keepdims=True)
    data_car = eeg_channels - mean_across_channels
    # data = eeg_channels[750:,:]   # after 3 seconds
    data = data_car  # after 3 seconds
    fs = 250  # Sample rate, Hz
    lowcut = 1
    highcut = 60
    notch_freq = 50  # Frequency to be notched out

    # Apply filters
    filtered_data = np.apply_along_axis(butter_bandpass_filter, 0, data, lowcut, highcut, fs)
    filtered_data = np.apply_along_axis(apply_notch_filter, 0, filtered_data, notch_freq, fs)

    # ---------------------- EEG Bands ---------------------

    # ---------------------- EEG Bands ---------------------

    # EEG frequency bands
    bands = {
        # 'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 60)  
    }
    band_power = {band: [] for band in bands}

    # Calculate PSD for each band 
    for i in range(filtered_data.shape[1]):
        freqs, psd = welch(filtered_data[:, i], fs, nperseg=128)
        for band, (low, high) in bands.items():
            # Find intersecting values
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            band_power[band].append(psd[idx_band].sum())  # Sum PSD within the band

    average_band_power_without_norm = {band: np.mean(powers) for band, powers in band_power.items()}

    return average_band_power_without_norm