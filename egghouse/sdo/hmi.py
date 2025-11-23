import numpy as np


def hmi_intscale(data):
    data = np.asarray(data, dtype=np.float64)
    data = (data + 100.) * (255./200.)
    image = np.clip(data, 0, 255).astype(np.uint8)
    return image