import tifffile as tiff
import matplotlib.pyplot as plt
from enum import Enum

# Worldview-3 - 8 Multispectral:
# Coastal: 400 - 450 nm (0, QGIS: 1, WV-3-Band-no:2)     Red: 630 - 690 nm       (4, QGIS: 5, WV-3-Band-no:6)
# Blue: 450 - 510 nm    (1, QGIS: 2, WV-3-Band-no:3)     Red Edge: 705 - 745 nm  (5, QGIS: 6, WV-3-Band-no:7)
# Green: 510 - 580 nm   (2, QGIS: 3, WV-3-Band-no:4)     Near-IR1: 770 - 895 nm  (6, QGIS: 7, WV-3-Band-no:8)
# Yellow: 585 - 625 nm  (3, QGIS: 4, WV-3-Band-no:5)     Near-IR2: 860 - 1040 nm (7, QGIS: 8, WV-3-Band-no:9)

# imshow RGB =  array 4, 2, 1
# QGIS RGB = bands 5, 3, 2. Contrast enhancement = Stretch to MinMax and carry out Load min/max values

class WV3ms(Enum):
    COASTAL = 0
    BLUE = 1
    GREEN = 2
    YELLOW = 3
    RED = 4
    REDEDGE = 5
    NIR1 = 6
    NIR2 = 7

m = tiff.imread('sixteen_band/6080_4_3_M.tif')

fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(10, 6))
ax = axes.ravel()

for spectrum in WV3ms:
    ax[spectrum.value].imshow(m[spectrum.value, :, :])
    ax[spectrum.value].set_title(spectrum.name)
    ax[spectrum.value].axis('off')

plt.tight_layout()
plt.show()
