import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


def stretchToMinMax(bands, lower_percent=2, higher_percent=98):
    # contrast enhancement as per QGIS Stretch to MinMax
    # and rescale in 0 .. 255
    out = np.zeros_like(bands)
    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[i, :, :], lower_percent)
        d = np.percentile(bands[i, :, :], higher_percent)
        t = a + (bands[i, :, :] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[i, :, :] = t
    return out

# tifffile RGB = ndarray shape (3, 3350, 3338) i.e. (colour, row, col)
# [0] = red, [1] = green, [2] = blue, 16 bit depth
tifffile_rgb = tiff.imread('three_band/6070_2_3.tif')

# display
fig, axes_subplot, axes_image = tiff.imshow(stretchToMinMax(tifffile_rgb), title="RGB", bitspersample=16, photometric='rgb')

axes_image.axes.axis('off')
plt.show()
