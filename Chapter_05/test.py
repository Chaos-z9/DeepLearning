import openvino as ov

core = ov.Core()
available_devices = core.available_devices
#available_devices = core.get_available_devices()
print(f"\n检测到可用硬件: {available_devices}")
print(ov.__version__)
