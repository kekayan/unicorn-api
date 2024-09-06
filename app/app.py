import logging
from contextlib import contextmanager
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect, WebSocketState
import asyncio
import threading
from queue import Queue
import json
import numpy as np

from .unicorn import unicorn
from .eeg_utils import extract_eeg_bands

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device_serial = None

app = FastAPI()

UNICORN_DEVICE_SERIAL_ID = "UN-2022.03.09"

def eeg_band_power_extract(data, sampling_freq=250):
    data = data[:, :8]
    data = np.asarray(data)
    extracted_features = extract_eeg_bands(data)
    logger.info(f"Extracted features: {extracted_features}")
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
    logger.info("Scanning for devices...")

    available_devices = unicorn.get_available_devices()

    if not available_devices:
        logger.warning("No devices found")
        return None

    logger.info(f"Available Devices: {available_devices}")

    if UNICORN_DEVICE_SERIAL_ID not in available_devices:
        logger.error(f"Device with serial {UNICORN_DEVICE_SERIAL_ID} not found")
        return None

    return UNICORN_DEVICE_SERIAL_ID

def acquire_data_thread(device, data_queue):
    try:
        logger.info("Starting acquisition...")
        unicorn.start_acquisition(device, True)
        buffer = []
        while True:
            data = unicorn.get_data(device, 1)
            buffer.append(data)
            if len(buffer) == 1000:
                eeg_bands = eeg_band_power_extract(np.array(buffer))
                data_queue.put(json.dumps(eeg_bands))
                logger.debug("Data sent to queue")
                buffer.clear()
                buffer = []
    except Exception as e:
        logger.exception(f"Error during data acquisition: {e}")
    finally:
        logger.debug("Stopping acquisition...")
        try:
            unicorn.stop_acquisition(device)
            logger.debug("Acquisition stopped")
        except Exception as e:
            logger.error(f"Error stopping acquisition: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type": "connected"
    })

    device_serial = scan_and_connect()

    if not device_serial:
        await websocket.close(code=1000)
        return
    try:
        with unicorn_device(device_serial) as device:
            logger.info(f"Connected to {device_serial}")

            data_queue = Queue()
            acquisition_thread = threading.Thread(target=acquire_data_thread, args=(device, data_queue))
            acquisition_thread.start()

            try:
                while True:
                    if not data_queue.empty():
                        data = data_queue.get()
                        await websocket.send_json({
                            "data": data,
                            "type": "band_power"
                        })
                        logger.debug("Data sent to client")
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
