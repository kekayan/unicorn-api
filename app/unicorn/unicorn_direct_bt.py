import serial
import time
import struct
import string
import random
import numpy as np
import mne
from scipy.signal import butter, lfilter, iirnotch, filtfilt,welch
import matplotlib.pyplot as plt


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


device='/dev/rfcomm0'
# UN-2022.03.09
blocksize=0.2
timeout=5
nchan=16
fsample=250

start_acq      = [0x61, 0x7C, 0x87]
stop_acq       = [0x63, 0x5C, 0xC5]
start_response = [0x00, 0x00, 0x00]
stop_response  = [0x00, 0x00, 0x00]
start_sequence = [0xC0, 0x00]
stop_sequence  = [0x0D, 0x0A]

try:
    s = serial.Serial(device, 115200, timeout=timeout)
    print("connected to serial port " + device)
except:
    raise RuntimeError("cannot connect to serial port " + device)

lsl_name    = 'Unicorn'
lsl_type    = 'EEG'
lsl_format  = 'float32'
lsl_id      = ''.join(random.choice(string.digits) for i in range(6))
                 

# start the Unicorn data stream
s.write(start_acq)
    
response = s.read(3)
if response != b'\x00\x00\x00':
    raise RuntimeError("cannot start data stream")

print('started Unicorn')

try:
    buffer = []
    count = 0
    while True:
        dat = np.zeros(nchan)
        
        # read one block of data from the serial port
        payload = s.read(45)
        
        # check the start and end bytes
        if payload[0:2] != b'\xC0\x00':
            raise RuntimeError("invalid packet")
        if payload[43:45] != b'\x0D\x0A':
            raise RuntimeError("invalid packet")
    
        battery = 100*float(payload[2] & 0x0F)/15
    
        eeg = np.zeros(8)
        for ch in range(0,8):
            # unpack as a big-endian 32 bit signed integer
            eegv = struct.unpack('>i', b'\x00' + payload[(3+ch*3):(6+ch*3)])[0]
            # apply two’s complement to the 32-bit signed integral value if the sign bit is set
            if (eegv & 0x00800000):
                eegv = eegv | 0xFF000000
            eeg[ch] = float(eegv) * 4500000. / 50331642.
    
        accel = np.zeros(3)
        # unpack as a little-endian 16 bit signed integer
        accel[0] = float(struct.unpack('<h', payload[27:29])[0]) / 4096.
        accel[1] = float(struct.unpack('<h', payload[29:31])[0]) / 4096.
        accel[2] = float(struct.unpack('<h', payload[31:33])[0]) / 4096.
    
        gyro = np.zeros(3)
        # unpack as a little-endian 16 bit signed integer
        gyro[0] = float(struct.unpack('<h', payload[27:29])[0]) / 32.8
        gyro[1] = float(struct.unpack('<h', payload[29:31])[0]) / 32.8
        gyro[2] = float(struct.unpack('<h', payload[31:33])[0]) / 32.8
    
        counter = struct.unpack('<L', payload[39:43])[0]
    
        # collect the data that will be sent to LSL
        # dat[0:8]   = eeg
        # dat[8:11]  = accel
        # dat[11:14] = gyro
        # dat[14]    = battery
        # dat[15]    = counter
        # eeg = eeg.reshape(1, -1)
        # print(eeg)
        buffer.append(eeg)
       
        if len(buffer) == 1000:
            if count == 0:
                count += 1
                buffer = []
                continue
            buffer = np.array(buffer)
            print(buffer.shape)
            # plotting
            mean_across_channels = np.mean(buffer, axis=1, keepdims=True)
            data_car = buffer - mean_across_channels
            data = data_car
            fs = 250  # Sample rate, Hz
            lowcut = 1
            highcut = 60
            notch_freq = 50  # Frequency to be notched out

            # Apply filters
            filtered_data = np.apply_along_axis(butter_bandpass_filter, 0, data, lowcut, highcut, fs)
            filtered_data = np.apply_along_axis(apply_notch_filter, 0, filtered_data, notch_freq, fs)

            ch_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
            

            eeg = data_car.T



            info = mne.create_info(ch_names, 250, ch_types=["eeg"] * 8)
            raw = mne.io.RawArray(eeg, info)
            raw.set_montage('standard_1020', on_missing='warn')
            
            raw_tmp = raw.copy()
            raw_tmp.filter(l_freq=5, h_freq=None)
            
            ica = mne.preprocessing.ICA(n_components=0.999999,method="picard",fit_params={"extended": True}, random_state=1)
            ica.fit(raw_tmp)

            ica.plot_components(inst=raw_tmp,show=False).savefig(f'imgs/ica_components_{time.time()}.png')
            ica.plot_sources(inst=raw_tmp, show=False).savefig(f'imgs/ica_sources_{time.time()}.png')

            bands = {
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 60)  
            }
            band_power = {band: [] for band in bands}

            # Calculate PSD for each band
            for i in range(data.shape[1]):
                freqs, psd = welch(data[:, i], 250, nperseg=128)
                for band, (low, high) in bands.items():
                    # Find intersecting values
                    idx_band = np.logical_and(freqs >= low, freqs <= high)
                    band_power[band].append(psd[idx_band].sum())  # Sum PSD within the band

            fig, axes = plt.subplots(nrows=len(bands), figsize=(10, 10))

            for i, (band, powers) in enumerate(band_power.items()):
                axes[i].bar(ch_names, powers, color='skyblue')
                axes[i].set_ylabel(f'{band} band power')
            axes[-1].set_xlabel('Channels')
            plt.tight_layout()
            # plt.show()

            plt.savefig(f'imgs/band_power_{time.time()}.png')
            average_band_power_without_norm = {band: np.mean(powers) for band, powers in band_power.items()}

            # 2
            fig, axes = plt.subplots(ncols=len(bands), figsize=(12, 6), sharey=True)  # Share the y-axis across subplots

            colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum']
            max_power = max(average_band_power_without_norm.values())

            for i, (band, powers) in enumerate(average_band_power_without_norm.items()):
                axes[i].bar(band, powers, color=colors[i % len(colors)])
                axes[i].set_ylim(0, max_power * 1.1) 
                axes[i].set_title(f'{band} Band Power')
                axes[i].grid(True, which='both', axis='y')  

            axes[-1].set_xlabel('Bands')
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'imgs/band_power_avg_{time.time()}.png')

            # send the data to file
            # with open('data_new_unicorn_direct.txt', 'a') as f:
            #     np.savetxt(f, eeg)
            buffer = []

        if ((counter % fsample) == 0):
            print('received %d samples, battery %d %%' % (counter, battery))

except Exception as e:
    print(e)
    print('closing')
    s.write(stop_acq)
    s.close()