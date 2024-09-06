import ctypes
from ctypes import c_float,c_char, c_int, c_uint8, c_uint16, c_uint32, c_uint64, c_char_p, c_bool, POINTER, Structure
import os

# Load the Unicorn library
lib_path = os.path.join(os.path.dirname(__file__), 'libunicorn.so')
unicorn_lib = ctypes.CDLL(lib_path)

# Define constants
UNICORN_ERROR_SUCCESS = 0
UNICORN_ERROR_INVALID_PARAMETER = 1
UNICORN_ERROR_BLUETOOTH_INIT_FAILED = 2
UNICORN_SERIAL_LENGTH_MAX = 14
UNICORN_TOTAL_CHANNELS_COUNT = 17
UNICORN_EEG_CONFIG_INDEX = 0
UNICORN_ACCELEROMETER_CONFIG_INDEX = 8
UNICORN_GYROSCOPE_CONFIG_INDEX = 11
UNICORN_SAMPLING_RATE = 250
UNICORN_EEG_CHANNELS_COUNT = 8
UNICORN_ACCELEROMETER_CHANNELS_COUNT = 3
UNICORN_GYROSCOPE_CHANNELS_COUNT = 3
UNICORN_NUMBER_OF_DIGITAL_OUTPUTS = 8


# Define structures
class UnicornAmplifierChannel(Structure):
    _fields_ = [
        ("name", c_char * 32),
        ("unit", c_char * 32),
        ("range", c_float * 2),
        ("enabled", c_bool)
    ]

class UnicornAmplifierConfiguration(Structure):
    _fields_ = [
        ("Channels", UnicornAmplifierChannel * UNICORN_TOTAL_CHANNELS_COUNT)
    ]

class UnicornDeviceInformation(Structure):
    _fields_ = [
        ("numberOfEegChannels", c_uint16),
        ("serial", c_char * UNICORN_SERIAL_LENGTH_MAX),
        ("firmwareVersion", c_char * 12),
        ("deviceVersion", c_char * 6),
        ("pcbVersion", c_uint8 * 4),
        ("enclosureVersion", c_uint8 * 4)
    ]

# Define function prototypes
unicorn_lib.UNICORN_GetApiVersion.restype = c_float
unicorn_lib.UNICORN_GetLastErrorText.restype = c_char_p

unicorn_lib.UNICORN_GetAvailableDevices.argtypes = [POINTER(c_char * UNICORN_SERIAL_LENGTH_MAX), POINTER(c_uint32), c_bool]
unicorn_lib.UNICORN_GetAvailableDevices.restype = c_int

unicorn_lib.UNICORN_OpenDevice.argtypes = [c_char_p, POINTER(c_uint64)]
unicorn_lib.UNICORN_OpenDevice.restype = c_int

unicorn_lib.UNICORN_CloseDevice.argtypes = [POINTER(c_uint64)]
unicorn_lib.UNICORN_CloseDevice.restype = c_int

unicorn_lib.UNICORN_StartAcquisition.argtypes = [c_uint64, c_bool]
unicorn_lib.UNICORN_StartAcquisition.restype = c_int

unicorn_lib.UNICORN_StopAcquisition.argtypes = [c_uint64]
unicorn_lib.UNICORN_StopAcquisition.restype = c_int

unicorn_lib.UNICORN_SetConfiguration.argtypes = [c_uint64, POINTER(UnicornAmplifierConfiguration)]
unicorn_lib.UNICORN_SetConfiguration.restype = c_int

unicorn_lib.UNICORN_GetConfiguration.argtypes = [c_uint64, POINTER(UnicornAmplifierConfiguration)]
unicorn_lib.UNICORN_GetConfiguration.restype = c_int

unicorn_lib.UNICORN_GetData.argtypes = [c_uint64, c_uint32, POINTER(c_float), c_uint32]
unicorn_lib.UNICORN_GetData.restype = c_int

unicorn_lib.UNICORN_GetNumberOfAcquiredChannels.argtypes = [c_uint64, POINTER(c_uint32)]
unicorn_lib.UNICORN_GetNumberOfAcquiredChannels.restype = c_int

unicorn_lib.UNICORN_GetChannelIndex.argtypes = [c_uint64, c_char_p, POINTER(c_uint32)]
unicorn_lib.UNICORN_GetChannelIndex.restype = c_int

unicorn_lib.UNICORN_GetDeviceInformation.argtypes = [c_uint64, POINTER(UnicornDeviceInformation)]
unicorn_lib.UNICORN_GetDeviceInformation.restype = c_int

unicorn_lib.UNICORN_SetDigitalOutputs.argtypes = [c_uint64, c_uint8]
unicorn_lib.UNICORN_SetDigitalOutputs.restype = c_int

unicorn_lib.UNICORN_GetDigitalOutputs.argtypes = [c_uint64, POINTER(c_uint8)]
unicorn_lib.UNICORN_GetDigitalOutputs.restype = c_int

# Wrapper functions
def get_api_version():
    return unicorn_lib.UNICORN_GetApiVersion()

def get_last_error_text():
    return unicorn_lib.UNICORN_GetLastErrorText().decode('utf-8')

def get_available_devices(rescan=False):
    count = c_uint32(0)
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        # First call to get the number of available devices
        result = unicorn_lib.UNICORN_GetAvailableDevices(None, ctypes.byref(count), rescan)
        if result != UNICORN_ERROR_SUCCESS:
            raise Exception(f"Error getting available devices count: {get_last_error_text()}")

        # Allocate buffer based on the returned count
        devices = (c_char * UNICORN_SERIAL_LENGTH_MAX * count.value)()

        # Second call to actually get the devices
        result = unicorn_lib.UNICORN_GetAvailableDevices(devices, ctypes.byref(count), rescan)

        if result == UNICORN_ERROR_SUCCESS:
            return [bytes(device).decode('utf-8').rstrip('\x00') for device in devices]
        elif result == UNICORN_ERROR_INVALID_PARAMETER and "Device buffer too small" in get_last_error_text():
            # If the buffer was too small, increase the count and try again
            count.value += 5  # Increase by 5 or another suitable number
            retry_count += 1
        else:
            raise Exception(f"Error getting available devices: {get_last_error_text()}")

    raise Exception("Failed to get available devices after multiple attempts")

def open_device(serial):
    handle = c_uint64()
    result = unicorn_lib.UNICORN_OpenDevice(serial.encode('utf-8'), ctypes.byref(handle))
    if result != UNICORN_ERROR_SUCCESS:
        raise Exception(f"Error opening device: {get_last_error_text()}")
    return handle.value

def close_device(handle):
    result = unicorn_lib.UNICORN_CloseDevice(ctypes.byref(c_uint64(handle)))
    if result != UNICORN_ERROR_SUCCESS:
        raise Exception(f"Error closing device: {get_last_error_text()}")

def start_acquisition(handle, test_signal_enabled=False):
    result = unicorn_lib.UNICORN_StartAcquisition(
        c_uint64(handle),
        c_bool(test_signal_enabled)
    )
    if result != UNICORN_ERROR_SUCCESS:
        raise Exception(f"Error starting acquisition: {get_last_error_text()}")

def stop_acquisition(handle):
    result = unicorn_lib.UNICORN_StopAcquisition(handle)
    if result != UNICORN_ERROR_SUCCESS:
        raise Exception(f"Error stopping acquisition: {get_last_error_text()}")

def set_configuration(handle, configuration):
    result = unicorn_lib.UNICORN_SetConfiguration(handle, ctypes.byref(configuration))
    if result != UNICORN_ERROR_SUCCESS:
        raise Exception(f"Error setting configuration: {get_last_error_text()}")

def get_configuration(handle):
    configuration = UnicornAmplifierConfiguration()
    result = unicorn_lib.UNICORN_GetConfiguration(handle, ctypes.byref(configuration))
    if result != UNICORN_ERROR_SUCCESS:
        raise Exception(f"Error getting configuration: {get_last_error_text()}")
    return configuration

def get_data(handle, number_of_scans):
    num_channels = c_uint32()
    unicorn_lib.UNICORN_GetNumberOfAcquiredChannels(handle, ctypes.byref(num_channels))
    buffer_length = number_of_scans * num_channels.value
    data_buffer = (c_float * buffer_length)()
    result = unicorn_lib.UNICORN_GetData(handle, number_of_scans, data_buffer, buffer_length)
    if result != UNICORN_ERROR_SUCCESS:
        raise Exception(f"Error getting data: {get_last_error_text()}")
    return list(data_buffer)

def get_channel_index(handle, channel_name):
    index = c_uint32()
    result = unicorn_lib.UNICORN_GetChannelIndex(handle, channel_name.encode('utf-8'), ctypes.byref(index))
    if result != UNICORN_ERROR_SUCCESS:
        raise Exception(f"Error getting channel index: {get_last_error_text()}")
    return index.value

def get_device_information(handle):
    info = UnicornDeviceInformation()
    result = unicorn_lib.UNICORN_GetDeviceInformation(handle, ctypes.byref(info))
    if result != UNICORN_ERROR_SUCCESS:
        raise Exception(f"Error getting device information: {get_last_error_text()}")
    return info

def set_digital_outputs(handle, digital_outputs):
    result = unicorn_lib.UNICORN_SetDigitalOutputs(handle, digital_outputs)
    if result != UNICORN_ERROR_SUCCESS:
        raise Exception(f"Error setting digital outputs: {get_last_error_text()}")

def get_digital_outputs(handle):
    digital_outputs = c_uint8()
    result = unicorn_lib.UNICORN_GetDigitalOutputs(handle, ctypes.byref(digital_outputs))
    if result != UNICORN_ERROR_SUCCESS:
        raise Exception(f"Error getting digital outputs: {get_last_error_text()}")
    return digital_outputs.value
