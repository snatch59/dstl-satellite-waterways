import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.transform import resize
import pandas as pd
from enum import Enum

# Worldview-3 - 8 Multispectral:
# Coastal: 400 - 450 nm (0, QGIS: 1, WV-3-Band-no:2)     Red: 630 - 690 nm       (4, QGIS: 5, WV-3-Band-no:6)
# Blue: 450 - 510 nm    (1, QGIS: 2, WV-3-Band-no:3)     Red Edge: 705 - 745 nm  (5, QGIS: 6, WV-3-Band-no:7)
# Green: 510 - 580 nm   (2, QGIS: 3, WV-3-Band-no:4)     Near-IR1: 770 - 895 nm  (6, QGIS: 7, WV-3-Band-no:8)
# Yellow: 585 - 625 nm  (3, QGIS: 4, WV-3-Band-no:5)     Near-IR2: 860 - 1040 nm (7, QGIS: 8, WV-3-Band-no:9)


class WV3ms(Enum):
    COASTAL = 0
    BLUE = 1
    GREEN = 2
    YELLOW = 3
    RED = 4
    REDEDGE = 5
    NIR1 = 6
    NIR2 = 7


CCCI_THRESHOLD_RGB = 0.11
CCCI_THRESHOLD_MS = 0.35


def stretch_8bit(bands, lower_percent=2, higher_percent=98):
    # contrast enhancement as per QGIS Stretch to MinMax
    # note that input image range is 0 .. 1
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


def CCCI_index_rgb(msdata, rgbdata):
    RE = resize(msdata[WV3ms.REDEDGE.value, :, :], (rgbdata.shape[0], rgbdata.shape[1]), mode='constant')
    NIR = resize(msdata[WV3ms.NIR2.value, :, :], (rgbdata.shape[0], rgbdata.shape[1]), mode='constant')
    R = rgbdata[:, :, 0]
    # R = resize(rgbdata[:, :, 0], (rgbdata.shape[0], rgbdata.shape[1]))
    # Canopy Chlorophyll Content Index
    CCCI = ((NIR - RE) / (NIR + RE)) / ((NIR - R) / (NIR + R))
    return CCCI


def CCCI_index_ms(msdata, rgbdata):
    RE = resize(msdata[WV3ms.REDEDGE.value, :, :], (rgbdata.shape[0], rgbdata.shape[1]), mode='constant')
    NIR = resize(msdata[WV3ms.NIR2.value, :, :], (rgbdata.shape[0], rgbdata.shape[1]), mode='constant')
    R = resize(msdata[WV3ms.RED.value, :, :], (rgbdata.shape[0], rgbdata.shape[1]), mode='constant')
    # Canopy Chlorophyll Content Index
    CCCI = ((NIR - RE) / (NIR + RE)) / ((NIR - R) / (NIR + R))
    return CCCI


def NVWI_index(msdata, rgbdata):
    G = resize(msdata[WV3ms.GREEN.value, :, :], (rgbdata.shape[0], rgbdata.shape[1]), mode='constant')
    NIR = resize(msdata[WV3ms.NIR1.value, :, :], (rgbdata.shape[0], rgbdata.shape[1]), mode='constant')
    NVWI = (G - NIR) / (G + NIR)
    return NVWI


def display(IM_ID):
    # read rgb and m bands
    rgb = tiff.imread('three_band/{}.tif'.format(IM_ID))
    rgb = np.rollaxis(rgb, 0, 3)
    m = tiff.imread('sixteen_band/{}_M.tif'.format(IM_ID))

    # get our indices
    myCCCI_rgb = CCCI_index_rgb(m, rgb)
    myCCCI_ms = CCCI_index_ms(m, rgb)
    myNDWI = NVWI_index(m, rgb)

    # you can look on histogram and pick your favorite threshold value
    ccci_binary_rgb = (myCCCI_rgb > CCCI_THRESHOLD_RGB).astype(np.float32)
    ccci_binary_ms = (myCCCI_ms > CCCI_THRESHOLD_MS).astype(np.float32)

    fig, axes = plt.subplots(ncols=5, nrows=1, figsize=(16, 4))
    ax = axes.ravel()
    ax[0].imshow(stretch_8bit(rgb))
    ax[0].set_title('Image')
    ax[0].axis('off')
    ax[1].imshow(myCCCI_ms, vmin=-.5, vmax=.5)
    ax[1].set_title('CCCI')
    ax[1].axis('off')
    ax[2].imshow(ccci_binary_rgb, cmap='binary_r')
    ax[2].set_title('CCCI - RGB Red')
    ax[2].axis('off')
    ax[3].imshow(ccci_binary_ms, cmap='binary_r')
    ax[3].set_title('CCCI - Multispectral Red')
    ax[3].axis('off')
    ax[4].imshow(myNDWI)
    ax[4].set_title('NDWI')
    ax[4].axis('off')
    plt.tight_layout()
    plt.show()


data = pd.read_csv('train_wkt_v4.csv')
data = data[data.MultipolygonWKT != 'MULTIPOLYGON EMPTY']

# use training data images for waterway
for IMG_ID in data[data.ClassType == 7].ImageId:
    display(IMG_ID)

# test images
# take some pictures from test
waterway_test = ['6080_4_3', '6080_4_0',
                 '6080_1_3', '6080_1_1',
                 '6150_3_4', '6050_2_1']

for IMG_ID in waterway_test:
    display(IMG_ID)
