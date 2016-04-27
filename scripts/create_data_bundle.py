"""
Create the data/data.npz file
"""
import os
import numpy as np
import numpy.ma as ma
import pylab as pl
import matplotlib.cm as cm
import joblib
import skimage
import unidecode
import skimage.color as skcolor
import gdal

DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')

label_names = [
    'labels_1',
    'labels_2',
    'labels_3',
    'labels_4',
    'labels_5',
    'labels_6',
    'labels_full'
]

labels = {}
for lname in label_names:
    ldata = {}
    _d = np.load(os.path.join(DATADIR, 'npy', '%s.npz' % lname))
    labels[lname] = {k:_d[k] for k in _d.keys()}

img = pl.imread(os.path.join(DATADIR, 'qgis', '2013_10_08_rgb.tif'))
dsm_ds = gdal.Open(os.path.join(DATADIR, 'qgis', '2013_10_08_dsm.tif'))
dsm = dsm_ds.ReadAsArray()

data = {
    'labels': labels,
    'img' : img,
    'dsm' : dsm
}

joblib.dump(data, os.path.join(DATADIR, 'data.joblib'), compress=True)
