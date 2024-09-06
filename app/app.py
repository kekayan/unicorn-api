import logging
from contextlib import contextmanager, asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect, WebSocketState
import asyncio
import threading
from queue import Queue
import json
import numpy as np
from scipy import signal

from copy import deepcopy

from .unicorn import unicorn
from .eeg_lib import eeg_features_extract

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device_serial = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global device_serial
    device_serial = scan_and_connect()
    yield
    # Clean up and release the resources
    device_serial = None

app = FastAPI(lifespan=lifespan)

UNICORN_DEVICE_SERIAL_ID = "UN-2022.03.09"

def eeg_band_power_extract(data, sampling_freq=250):
    data = data[:, :8]
    data = np.asarray(data)
    extracted_features = eeg_features_extract(data, sampling_freq)
    print(extracted_features)
    # [alpha, beta, theta, gamma, delta]
    return extracted_features

    

@contextmanager
def unicorn_device(serial_id):
    device = None
    try:
        device = unicorn.open_device(serial_id)
        yield device
    finally:
        if device:
            unicorn.close_device(device)

def scan_and_connect():
    logger.info(f"Unicorn API Version: {unicorn.get_api_version()}")
    available_devices = unicorn.get_available_devices()

    logger.info("Scanning for devices...")
    if not available_devices:
        logger.warning("No devices found")
        return None

    logger.info(f"Available Devices: {available_devices}")

    if UNICORN_DEVICE_SERIAL_ID not in available_devices:
        logger.error(f"Device with serial {UNICORN_DEVICE_SERIAL_ID} not found")
        return None

    return UNICORN_DEVICE_SERIAL_ID

def acquire_data_thread(device, data_queue, raw_data_queue):
    try:
        logger.info("Starting acquisition...")
        unicorn.start_acquisition(device, True)
        buffer = []
        buffer2 = []
        while True:
            data = unicorn.get_data(device, 1)
            buffer.append(data)
            raw_data_queue.put(data)
           
            # print(data)
            if len(buffer) == 1000:
                
                
                if buffer2:
                    logger.info(".................")
                    logger.info(np.array(buffer2) - np.array(buffer))
                    logger.info(".................")
                    buffer2 = []
                else:
                    logger.info("buffer2 is empty, skipping subtraction")
              
                eeg_bands = eeg_band_power_extract(np.array(buffer))
                buffer2 = deepcopy(buffer)

                data_queue.put(json.dumps(eeg_bands))
                # logger.info("Data sent to queue")
                buffer.clear()
                buffer = []
    except Exception as e:
        logger.exception(f"Error during data acquisition: {e}")
    finally:
        # logger.info("Stopping acquisition...")
        try:
            unicorn.stop_acquisition(device)
            # logger.info("Acquisition stopped")
        except Exception as e:
            logger.error(f"Error stopping acquisition: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not device_serial:
        await websocket.close(code=1000, reason="No device found")
        return

    await websocket.accept()

    try:
        with unicorn_device(device_serial) as device:
            # logger.info(f"Connected to {device_serial}")

            data_queue = Queue()
            raw_data_queue = Queue()
            acquisition_thread = threading.Thread(target=acquire_data_thread, args=(device, data_queue,raw_data_queue))
            acquisition_thread.start()

            try:
                while True:
                    if not raw_data_queue.empty():
                        raw_data = raw_data_queue.get()
                        
                        # await websocket.send_json({
                        #     "raw_data": raw_data,
                        #     "type": "raw_data"
                        # })
                        # logger.info("Raw Data sent to client")
                    if not data_queue.empty():
                        data = data_queue.get()
                        await websocket.send_json({
                            "data": data,
                            "type": "band_power"
                        })
                        # logger.info("Data sent to client")
                    else:
                        await asyncio.sleep(0.01)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
            finally:
                # Stop the acquisition thread
                acquisition_thread.join(timeout=1)
                if acquisition_thread.is_alive():
                    logger.warning("Acquisition thread did not stop in time")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1000)
