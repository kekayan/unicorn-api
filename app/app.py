import os
import serial
import asyncio
import struct
import threading
import numpy as np
from scipy.signal import welch
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect, WebSocketState

from dotenv import load_dotenv

from .eeg_utils import butter_bandpass_filter, apply_notch_filter

from queue import Queue


from dotenv import load_dotenv

load_dotenv()


device='/dev/rfcomm0'

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




app = FastAPI()

async def run_sudo_command(*args):
       sudo_password = os.environ.get("SUDO_PASSWORD")
       if not sudo_password:
           raise ValueError("SUDO_PASSWORD environment variable is not set")

       cmd = " ".join(["sudo", "-S"] + list(args))
       process = await asyncio.create_subprocess_shell(
           cmd,
           stdin=asyncio.subprocess.PIPE,
           stdout=asyncio.subprocess.PIPE,
           stderr=asyncio.subprocess.PIPE,
           shell=True
       )

       stdout, stderr = await process.communicate(input=f"{sudo_password}\n".encode())

       if process.returncode != 0:
           raise RuntimeError(f"Command failed: {stderr.decode()}")

       return stdout.decode()

def data_acquisition(data_queue, stop_event):
    try:
        s = serial.Serial(device, 115200, timeout=timeout)
        print("connected to serial port " + device)
    except:
        raise RuntimeError("cannot connect to serial port " + device)

    # start the Unicorn data stream
    s.write(start_acq)
    buffer = []
    count = 0
    response = s.read(3)
    if response != b'\x00\x00\x00':
        raise RuntimeError("cannot start data stream")

    print('started Unicorn')
    data_queue.put({
        "type": "started"
    })

    while not stop_event.is_set():


        try:
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

            buffer.append(eeg)

            # accel = np.zeros(3)
            # # unpack as a little-endian 16 bit signed integer
            # accel[0] = float(struct.unpack('<h', payload[27:29])[0]) / 4096.
            # accel[1] = float(struct.unpack('<h', payload[29:31])[0]) / 4096.
            # accel[2] = float(struct.unpack('<h', payload[31:33])[0]) / 4096.

            # gyro = np.zeros(3)
            # # unpack as a little-endian 16 bit signed integer
            # gyro[0] = float(struct.unpack('<h', payload[27:29])[0]) / 32.8
            # gyro[1] = float(struct.unpack('<h', payload[29:31])[0]) / 32.8
            # gyro[2] = float(struct.unpack('<h', payload[31:33])[0]) / 32.8

            counter = struct.unpack('<L', payload[39:43])[0]

            # collect the data that will be sent to LSL
            # dat[0:8]   = eeg
            # dat[8:11]  = accel
            # dat[11:14] = gyro
            # dat[14]    = battery
            # dat[15]    = counter



            if len(buffer) == 1000:
                if count == 0:
                    count += 1
                    buffer = []
                    continue
                buffer = np.array(buffer)
                print("time to process")

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




                # eeg = data_car.T

                # ch_names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
                # info = mne.create_info(ch_names, 250, ch_types=["eeg"] * 8)
                # raw = mne.io.RawArray(eeg, info)
                # raw.set_montage('standard_1020', on_missing='warn')

                # raw_tmp = raw.copy()
                # raw_tmp.filter(l_freq=5, h_freq=None)

                # ica = mne.preprocessing.ICA(n_components=0.999999,method="picard",fit_params={"extended": True}, random_state=1)
                # ica.fit(raw_tmp)

                # ica.plot_components(inst=raw_tmp,show=False).savefig(f'imgs/ica_components_{time.time()}.png')
                # ica.plot_sources(inst=raw_tmp, show=False).savefig(f'imgs/ica_sources_{time.time()}.png')

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

                data_queue.put({
                    "type": "band_power",
                    "data": band_power
                })
                print("add band power")

                # fig, axes = plt.subplots(nrows=len(bands), figsize=(10, 10))

                # for i, (band, powers) in enumerate(band_power.items()):
                #     axes[i].bar(ch_names, powers, color='skyblue')
                #     axes[i].set_ylabel(f'{band} band power')
                # axes[-1].set_xlabel('Channels')
                # plt.tight_layout()
                # # plt.show()

                # plt.savefig(f'imgs/band_power_{time.time()}.png')
                average_band_power_without_norm = {band: np.mean(powers) for band, powers in band_power.items()}

                data_queue.put({
                    "type": "average_band_power",
                    "data": average_band_power_without_norm
                })
                print("add average band power")

                # 2
                # fig, axes = plt.subplots(ncols=len(bands), figsize=(12, 6), sharey=True)  # Share the y-axis across subplots

                # colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum']
                # max_power = max(average_band_power_without_norm.values())

                # for i, (band, powers) in enumerate(average_band_power_without_norm.items()):
                #     axes[i].bar(band, powers, color=colors[i % len(colors)])
                #     axes[i].set_ylim(0, max_power * 1.1)
                #     axes[i].set_title(f'{band} Band Power')
                #     axes[i].grid(True, which='both', axis='y')

                # axes[-1].set_xlabel('Bands')
                # plt.tight_layout()
                # # plt.show()
                # plt.savefig(f'imgs/band_power_avg_{time.time()}.png')

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

async def run_rfcomm():
       process = await asyncio.create_subprocess_exec(
           "sudo", "-S", "rfcomm", "connect", "/dev/rfcomm0", "84:2E:14:09:EC:73",
           stdin=asyncio.subprocess.PIPE,
           stdout=asyncio.subprocess.PIPE,
           stderr=asyncio.subprocess.PIPE
       )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type": "connected"
    })
    try:
        # sudo rfcomm release /dev/rfcomm0
        await websocket.send_json({
            "type": "setting_up_device",
            "message": "Releasing rfcomm0"
        })
        await run_sudo_command("rfcomm", "release", "/dev/rfcomm0")
        await websocket.send_json({
            "type": "setting_up_device",
            "message": "Released rfcomm0"
        })
        # Run the rfcomm connect command
        await websocket.send_json({
            "type": "setting_up_device",
            "message": "Connecting to rfcomm"
        })
        print('connecting to rfcomm')
        rfcomm_task = asyncio.create_task(run_rfcomm())
        await asyncio.sleep(5)  # Wait for 5 seconds
        if os.path.exists("/dev/rfcomm0"):
            await run_sudo_command("chmod", "666", "/dev/rfcomm0")
        else:
            print("Device file /dev/rfcomm0 does not exist")
            return
        print('connected to rfcomm')
        await websocket.send_json({
            "type": "setting_up_device",
            "message": "Connected to rfcomm"
        })

        # Run the chmod command
        print('chmodding')
        await websocket.send_json({
            "type": "setting_up_device",
            "message": "Changing permissions on rfcomm0"
        })
        await run_sudo_command("chmod", "666", "/dev/rfcomm0")
        print('chmodded')
        await websocket.send_json({
            "type": "setting_up_device",
            "message": "Changed permissions on rfcomm0"
        })
    except Exception as e:
        print(f"Error setting up device: {e}")
        await websocket.close(code=1000)
        return

    data_queue = Queue()
    stop_event = threading.Event()
    acquisition_thread = threading.Thread(target=data_acquisition, args=(data_queue, stop_event))
    acquisition_thread.start()

    websocket_closed = False
    await websocket.send_json({
        "type": "acquisition_started"
    })

    try:
        while True:
            if not data_queue.empty():
                print('sending data')
                data = data_queue.get()
                await websocket.send_json(data)
                print('sent data')

            # Add a small delay to prevent busy-waiting
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print('websocket disconnected')
        websocket_closed = True
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print('closing')
        stop_event.set()
        if acquisition_thread and acquisition_thread.is_alive():
            acquisition_thread.join(timeout=5)
        # Disconnect rfcomm
        try:
            await run_sudo_command("rfcomm", "release", "/dev/rfcomm0")
        except Exception as e:
            print(f"Error disconnecting rfcomm: {e}")

        if not websocket_closed and websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1000)
