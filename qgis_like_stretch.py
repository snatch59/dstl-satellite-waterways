import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


def stretch_8bit(bands, lower_percent=2, higher_percent=98):
    # contrast enhancement as per QGIS Stretch to MinMax
    # and rescale in 0 .. 255
    out = np.zeros_like(bands)
    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.uint8)


def stretch_8bit2(bands, lower_percent=2, higher_percent=98):
    # contrast enhancement as per QGIS Stretch to MinMax
    # and rescale in 0 .. 1
    out = np.zeros_like(bands).astype(np.float32)
    for i in range(3):
        a = 0
        b = 1
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.float32)


# tifffile RGB = ndarray shape (3, 3350, 3338) i.e. (colour, row, col)
# [0] = red, [1] = green, [2] = blue, 16 bit depth
tifffile_rgb = tiff.imread('three_band/6070_2_3.tif')

# change shape to regular (3350, 3338, 3) i.e. (row, col, colour)
rgb_regular = np.rollaxis(tifffile_rgb, 0, 3)

# display
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(stretch_8bit(rgb_regular))
ax[0].set_title('RGB')
ax[0].axis('off')

ax[1].imshow(stretch_8bit2(rgb_regular))
ax[1].set_title('RGB')
ax[1].axis('off')

plt.show()
