import numpy as np
from scipy import ndimage
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
from skimage.transform import resize

# Worldview-3 - Panchromatic (3349, 3338): 400nm - 800nm

# Worldview-3 RGB (3350, 3338)

# Worldview-3 - 8 Multispectral bands (838, 835):
# Coastal: 400 - 450 nm (0, QGIS: 1, WV-3-Band-no:2)     Red: 630 - 690 nm       (4, QGIS: 5, WV-3-Band-no:6)
# Blue: 450 - 510 nm    (1, QGIS: 2, WV-3-Band-no:3)     Red Edge: 705 - 745 nm  (5, QGIS: 6, WV-3-Band-no:7)
# Green: 510 - 580 nm   (2, QGIS: 3, WV-3-Band-no:4)     Near-IR1: 770 - 895 nm  (6, QGIS: 7, WV-3-Band-no:8)
# Yellow: 585 - 625 nm  (3, QGIS: 4, WV-3-Band-no:5)     Near-IR2: 860 - 1040 nm (7, QGIS: 8, WV-3-Band-no:9)

# NIR - Near Infra Red: 750nm - 1400nm
# MIR - Mid Infra Red: 3000nm - 8000nm

# Worldview-3 - 8 SWIR bands (134, 133):
# SWIR-1: 1195 - 1225 nm    SWIR-5: 2145 - 2185 nm
# SWIR-2: 1550 - 1590 nm    SWIR-6: 2185 - 2225 nm
# SWIR-3: 1640 - 1680 nm    SWIR-7: 2235 - 2285 nm
# SWIR-4: 1710 - 1750 nm    SWIR-8: 2295 - 2365 nm


class WV3ms(Enum):
    COASTAL = 0
    BLUE = 1
    GREEN = 2
    YELLOW = 3
    RED = 4
    REDEDGE = 5
    NEARIR1 = 6
    NEARIR2 = 7


class WV3swir(Enum):
    SWIR_1 = 0
    SWIR_2 = 1
    SWIR_3 = 2
    SWIR_4 = 3
    SWIR_5 = 4
    SWIR_6 = 5
    SWIR_7 = 6
    SWIR_8 = 7


CCCI_THRESHOLD_U = 0.5
CCCI_THRESHOLD_L = -4
FAUX_CCCI_THRESHOLD = 0.11
# CCCI_SWIR_THRESHOLD = 1.03
CCCI_SWIR_THRESHOLD = .94
NDWI_THRESHOLD = 0.07
NDVI_THRESHOLD = 0.07


def stretch_8bit(bands, lower_percent=2, higher_percent=98, depth=3):
    # contrast enhancement as per QGIS Stretch to MinMax
    # note that input image range is 0 .. 1
    out = np.zeros_like(bands).astype(np.float32)
    for i in range(depth):
        a = 0
        b = 1
        if depth == 1:
            c = np.percentile(bands[:, :], lower_percent)
            d = np.percentile(bands[:, :], higher_percent)
            t = a + (bands[:, :] - c) * (b - a) / (d - c)
        else:
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], higher_percent)
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        if depth == 1:
            out[:, :] = t
        else:
            out[:, :, i] = t
    return out.astype(np.float32)


def EVI_index(msdata):
    # Enhanced Vegetation Index
    NIR2 = msdata[WV3ms.NEARIR2.value, :, :].astype(np.float32)
    R = msdata[WV3ms.RED.value, :, :].astype(np.float32)
    CB = msdata[WV3ms.COASTAL.value, :, :].astype(np.float32)

    # EVI = 2.5 * (NIR2 - R)/(NIR2 + 6.0*R - 7.5*CB + 1.0)
    a = 2.5 * (NIR2 - R)
    b = NIR2 + 6.0*R - 7.5*CB + 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        EVI = np.true_divide(a, b)
        EVI[EVI == np.inf] = 0
        EVI = np.nan_to_num(EVI)
    return EVI


def SAVI_index(msdata):
    # Soil Adjusted Vegetation Index
    NIR1 = msdata[WV3ms.NEARIR1.value, :, :].astype(np.float32)
    R = msdata[WV3ms.RED.value, :, :].astype(np.float32)
    # The value of L varies by the amount or cover of green vegetation: in very high vegetation regions,
    # L=0; and in areas with no green vegetation, L=1. Generally, an L=0.5 works well in most situations
    # and is the default value used. When L=0, then SAVI = NDVI.
    L = 0.5

    # SAVI = (1 + L) * (NIR1 - R)/(NIR1 + R + L)
    a = (1 + L) * (NIR1 - R)
    b = NIR1 + R + L
    with np.errstate(divide='ignore', invalid='ignore'):
        SAVI = np.true_divide(a, b)
        SAVI[SAVI == np.inf] = 0
        SAVI = np.nan_to_num(SAVI)
    return SAVI


def faux_CCCI_index(msdata, rgbdata):
    RE = resize(msdata[WV3ms.REDEDGE.value, :, :], (rgbdata.shape[0], rgbdata.shape[1]),
                mode='constant', preserve_range=False)
    NIR2 = resize(msdata[WV3ms.NEARIR2.value, :, :], (rgbdata.shape[0], rgbdata.shape[1]),
                  mode='constant', preserve_range=False)
    R = rgbdata[:, :, 0]

    # resize: note that with the default preserve_range=False the input image is
    # converted according to the conventions of img_as_float (values in [0, 1])
    # from the original 11 bits range [0, 2047]. preserve_range=True should be used.
    # faux_CCCI_index only works preserve_range=False - reason unknown

    # Canopy Chlorophyll Content Index
    # CCCI = ((NIR2 - RE) / (NIR2 + RE)) / ((NIR2 - R) / (NIR2 + R))
    a = NIR2 - RE
    b = NIR2 + RE
    # c = NIR2 - R
    # d = NIR2 + R
    c = R * (-1)
    d = R

    with np.errstate(divide='ignore', invalid='ignore'):
        e = np.true_divide(a, b)
        e[e == np.inf] = 0
        e = np.nan_to_num(e)
        f = np.true_divide(c, d)
        f[f == np.inf] = 0
        f = np.nan_to_num(f)
        CCCI = np.true_divide(e, f)
        CCCI[CCCI == np.inf] = 0
        CCCI = np.nan_to_num(CCCI)
    return CCCI


def CCCI_NIR2_index(msdata):
    # Canopy Chlorophyll Content Index
    # uses NIR2 rather than SWIR_1
    RE = msdata[WV3ms.REDEDGE.value, :, :].astype(np.float32)
    NIR2 = msdata[WV3ms.NEARIR2.value, :, :].astype(np.float32)
    R = msdata[WV3ms.RED.value, :, :].astype(np.float32)

    # CCCI = ((NIR2 - RE)/ NIR2 + RE)) / ((NIR2 - R)/(NIR2 + R))
    a = NIR2 - RE
    b = NIR2 + RE
    c = NIR2 - R
    d = NIR2 + R

    with np.errstate(divide='ignore', invalid='ignore'):
        e = np.true_divide(a, b)
        e[e == np.inf] = 0
        e = np.nan_to_num(e)
        f = np.true_divide(c, d)
        f[f == np.inf] = 0
        f = np.nan_to_num(f)
        CCCI = np.true_divide(e, f)
        CCCI[CCCI == np.inf] = 0
        CCCI = np.nan_to_num(CCCI)
    return CCCI


def CCCI_SWIR_index(msdata, swirdata):
    # Canopy Chlorophyll Content Index
    # uses SWIR_1
    RE = msdata[WV3ms.REDEDGE.value, :, :].astype(np.float32)
    SWIR1 = resize(swirdata[WV3swir.SWIR_1.value, :, :], (msdata.shape[1], msdata.shape[2]),
                  mode='constant', preserve_range=True).astype(np.float32)
    R = msdata[WV3ms.RED.value, :, :].astype(np.float32)

    # CCCI = ((SWIR1 - RE)/ SWIR1 + RE)) / ((SWIR1 - R)/(SWIR1 + R))
    a = SWIR1 - RE
    b = SWIR1 + RE
    c = SWIR1 - R
    d = SWIR1 + R

    with np.errstate(divide='ignore', invalid='ignore'):
        e = np.true_divide(a, b)
        e[e == np.inf] = 0
        e = np.nan_to_num(e)
        f = np.true_divide(c, d)
        f[f == np.inf] = 0
        f = np.nan_to_num(f)
        CCCI = np.true_divide(e, f)
        CCCI[CCCI == np.inf] = 0
        CCCI = np.nan_to_num(CCCI)
    return CCCI


def NDWI_index(msdata):
    # Normalized Difference Water Index
    # Uses McFeeter's NDWI based on MODIS band 2 and band 4
    G = msdata[WV3ms.GREEN.value, :, :].astype(np.float32)
    NIR1 = msdata[WV3ms.NEARIR1.value, :, :].astype(np.float32)

    # NDWI = (G - NIR1)/(G + NIR1)
    a = G - NIR1
    b = G + NIR1
    with np.errstate(divide='ignore', invalid='ignore'):
        NDWI = np.true_divide(a, b)
        NDWI[NDWI == np.inf] = 0
        NDWI = np.nan_to_num(NDWI)
    return NDWI


def NDVI_index(msdata):
    # Normalized Difference Vegetation Index
    R = msdata[WV3ms.RED.value, :, :].astype(np.float32)
    NIR1 = msdata[WV3ms.NEARIR1.value, :, :].astype(np.float32)

    # NDVI = (NIR1 - R)/(NIR1 + R )
    a = NIR1 - R
    b = NIR1 + R
    with np.errstate(divide='ignore', invalid='ignore'):
        NDVI = np.true_divide(a, b)
        NDVI[NDVI == np.inf] = 0
        NDVI = np.nan_to_num(NDVI)
    return NDVI

def display(IM_ID):
    # read rgb and m bands

    # tifffile RGB = ndarray shape (3, 3350, 3338) i.e. (colour, row, col)
    # [0] = red, [1] = green, [2] = blue, 16 bit depth
    rgb = tiff.imread('three_band/{}.tif'.format(IM_ID))
    # change shape to regular (3350, 3338, 3) i.e. (row, col, colour)
    rgb = np.rollaxis(rgb, 0, 3)

    # tifffile M = ndarray shape (8, 838, 835) i.e. (spectrum, row, col)
    m = tiff.imread('sixteen_band/{}_M.tif'.format(IM_ID))

    # tiffile panchrom = ndarray shape (3349, 3338) i.e. (row, col)
    panchrom = tiff.imread('sixteen_band/{}_P.tif'.format(IM_ID))

    # tiffile SWIR = ndarray shape (8, 134, 133) i.e. (spectrum, row, col)
    swir = tiff.imread('sixteen_band/{}_A.tif'.format(IM_ID))

    # get our indices
    myFauxCCCI = faux_CCCI_index(m, rgb)
    myCCCI = CCCI_NIR2_index(m)
    mySwirCCCI = CCCI_SWIR_index(m, swir)
    myNDWI = NDWI_index(m)
    myNDVI = NDVI_index(m)
    myEVI = EVI_index(m)
    mySAVI = SAVI_index(m)

    # you can look on histogram and pick your favorite threshold value
    # ccci_binary = (myCCCI < CCCI_THRESHOLD).astype(np.float32)
    ccci_binary_1 = (myCCCI < CCCI_THRESHOLD_U)
    ccci_binary_2 = (myCCCI > CCCI_THRESHOLD_L)
    ccci_binary_3 = np.logical_and(ccci_binary_1, ccci_binary_2)
    ccci_binary_4 = np.logical_not(ccci_binary_3)
    ccci_binary_5 = ndimage.binary_opening(ccci_binary_4)
    ccci_binary = ndimage.binary_closing(ccci_binary_5).astype(np.float32)

    ndwi_binary = (myNDWI > NDWI_THRESHOLD).astype(np.float32)
    ndvi_binary = (myNDWI > NDVI_THRESHOLD).astype(np.float32)
    faux_ccci_binary = (myFauxCCCI > FAUX_CCCI_THRESHOLD).astype(np.float32)
    ccci_swir_binary = (mySwirCCCI > CCCI_SWIR_THRESHOLD).astype(np.float32)

    fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(18, 9))
    ax = axes.ravel()
    ax[0].imshow(ccci_binary, cmap='binary_r')
    ax[0].set_title('CCCI NIR 2 Mask')
    ax[0].axis('off')
    ax[1].imshow(ndwi_binary, cmap='binary_r')
    ax[1].set_title('NDWI Mask')
    ax[1].axis('off')
    ax[2].imshow(ndvi_binary, cmap='binary_r')
    ax[2].set_title('NDVI Mask')
    ax[2].axis('off')
    ax[3].imshow(faux_ccci_binary, cmap='binary_r')
    ax[3].set_title('Faux CCCI Mask')
    ax[3].axis('off')
    ax[4].imshow(ccci_swir_binary, cmap='binary_r')
    ax[4].set_title('CCCI SWIR 1 Mask')
    ax[4].axis('off')

    hist, bins = np.histogram(myCCCI, range=(-2, 2), bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax[5].set_title('CCCI NIR 2 Histogram')
    ax[5].bar(center, hist, align='center', width=width)

    hist, bins = np.histogram(myNDWI, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax[6].set_title('NDWI Histogram')
    ax[6].bar(center, hist, align='center', width=width)

    hist, bins = np.histogram(myNDVI, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax[7].set_title('NDVI Histogram')
    ax[7].bar(center, hist, align='center', width=width)

    hist, bins = np.histogram(myFauxCCCI, range=(-.4, .4), bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax[8].set_title('Faux CCCI Histogram')
    ax[8].bar(center, hist, align='center', width=width)

    hist, bins = np.histogram(mySwirCCCI, range=(.4, 1.2), bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax[9].set_title('CCCI SWIR 1 Histogram')
    ax[9].bar(center, hist, align='center', width=width)

    plt.tight_layout()
    plt.show()

    # fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 10))
    # ax = axes.ravel()
    # ax[0].imshow(stretch_8bit(rgb))
    # ax[0].set_title('RGB {}'.format(IM_ID))
    # ax[0].axis('off')
    # ax[1].imshow(stretch_8bit(panchrom, depth=1), cmap='gray')
    # ax[1].set_title('Panchromatic {}'.format(IM_ID))
    # ax[1].axis('off')
    # plt.tight_layout()
    # plt.show()

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(18, 10))
    ax = axes.ravel()
    ax[0].imshow(myCCCI, vmin=-.5, vmax=.5)
    ax[0].set_title('CCCI NIR 2')
    ax[0].axis('off')
    ax[1].imshow(myNDWI, vmin=-.3, vmax=.3)
    ax[1].set_title('NDWI')
    ax[1].axis('off')
    ax[2].imshow(myNDVI)
    ax[2].set_title('NDVI')
    ax[2].axis('off')
    ax[3].imshow(myEVI, vmin=-.5, vmax=.5)
    ax[3].set_title('EVI')
    ax[3].axis('off')
    ax[4].imshow(mySAVI)
    ax[4].set_title('SAVI')
    ax[4].axis('off')
    ax[5].imshow(mySwirCCCI, vmin=0.6, vmax=1.2)
    ax[5].set_title('CCCI SWIR 1')
    ax[5].axis('off')
    plt.tight_layout()
    plt.show()


# -----Main------
data = pd.read_csv('train_wkt_v4.csv')
data = data[data.MultipolygonWKT != 'MULTIPOLYGON EMPTY']

# display('6150_3_4')

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
