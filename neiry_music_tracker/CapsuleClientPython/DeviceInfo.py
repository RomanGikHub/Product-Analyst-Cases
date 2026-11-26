import ctypes

from DeviceType import DeviceType

class DeviceInfo:
    def __init__(self, pointer, lib):
        self._pointer = pointer
        self._lib = lib
        self._lib.clCDeviceInfo_GetName.restype = ctypes.c_char_p
        self._lib.clCDeviceInfo_GetSerial.restype = ctypes.c_char_p
        self._lib.clCDeviceInfo_GetType.restype = ctypes.c_int
        self._lib.clCDeviceInfo_GetName.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._name = self._lib.clCDeviceInfo_GetName(self._pointer).decode('utf-8')
        self._serial = self._lib.clCDeviceInfo_GetSerial(self._pointer).decode('utf-8')
        self._type = DeviceType(self._lib.clCDeviceInfo_GetType(self._pointer))

    def get_name(self):
        return self._name

    def get_serial(self):
        return self._serial

    def get_type(self):
        return self._type.value
