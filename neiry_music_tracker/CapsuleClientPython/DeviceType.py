import ctypes

class DeviceType(ctypes.c_int):
    Headband = 0
    Buds = 1
    Headphones = 2
    Impulse = 3
    Any = 4
    BrainBit = 6
    SinWave = 100
    Noise = 101
