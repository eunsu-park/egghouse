import numpy as np
from ..image import bytescale_image


def aia_intscale(image, exptime=None, wavelnth=None, bytescale=False):

    image[np.isnan(image)] = 0.

    wavelnth = np.rint(wavelnth)
    
    if wavelnth == 94 :
        vmin, vmax = 1.5 / 1.06, 50 / 1.06
        temp = image * (4.99803 / exptime)

    elif wavelnth == 131 :
        vmin, vmax = 7.0 / 1.49, 1200 / 1.49
        temp = image * (6.99685 / exptime)

    elif wavelnth == 171 :
        vmin, vmax = 10.0 / 1.49, 6000 / 1.49
        temp = image * (4.99803 / exptime)

    elif wavelnth == 193 :
        vmin, vmax = 120.0 / 2.2, 6000.0 / 2.2
        temp = image * (2.9995 / exptime)

    elif wavelnth == 211 :
        vmin, vmax = 30.0 / 1.10, 13000 / 1.10
        temp = image * (4.99801 / exptime)

    elif wavelnth == 304 :
        vmin, vmax = 50.0 / 12.11, 2000 / 12.11
        temp = image * (4.99941 / exptime)

    elif wavelnth == 335 :
        vmin, vmax = 3.5 / 2.97, 1000 / 2.97
        temp = image * (6.99734 / exptime)

    elif wavelnth == 1600 :
        vmin, vmax = -8, 200
        temp = image * (2.99911 / exptime)

    elif wavelnth == 1700 :
        vmin, vmax = 0, 2500
        temp = image * (1.00026 / exptime)

    elif wavelnth == 4500 :
        vmin, vmax = 0, 26000
        temp = image * (1.00026 / exptime)

    elif wavelnth == 6173 :
        vmin, vmax = 0, 65535
        temp = image / exptime

    temp = np.clip(temp, vmin, vmax)
    if wavelnth in (94, 171) :
        scaled = bytescale_image(np.sqrt(temp), np.sqrt(vmin), np.sqrt(vmax))
    elif wavelnth in (131, 193, 211, 304, 335) :
        scaled = bytescale_image(np.log10(temp), np.log10(vmin), np.log10(vmax))
    elif wavelnth in (1600, 1700, 4500, 6173):
        scaled = bytescale_image(temp, vmin, vmax)

    return scaled
