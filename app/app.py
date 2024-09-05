
import logging
from contextlib import contextmanager, asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect, WebSocketState
import asyncio

from .unicorn import unicorn

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not device_serial:
        await websocket.close(code=1000, reason="No device found")
        return

    await websocket.accept()

    try:
        with unicorn_device(device_serial) as device:
            logger.info(f"Connected to {device_serial}")
            await acquire_data(device, websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1000)

async def acquire_data(device, websocket: WebSocket):
    try:
        logger.info("Starting acquisition...")
        unicorn.start_acquisition(device, True)

        while True:
            data = unicorn.get_data(device, 1)
            await websocket.send_json(data)
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected during data acquisition")
    except Exception as e:
        logger.exception(f"Error during data acquisition: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1000)
    finally:
        logger.info("Stopping acquisition...")
        try:
            unicorn.stop_acquisition(device)
            logger.info("Acquisition stopped")
        except Exception as e:
            logger.error(f"Error stopping acquisition: {e}")
