# Utilities to deal with the classif2 agrovision dataset
import numpy as np
import numpy.ma as ma
import os
import gdal

BASEDIR = os.path.join(os.environ['AGRODATA'], '3_datasets', 'classif2')

def load_labels(num=1):
    """
    Loads the classif2 labels
    Args:
        num: The number of the labels set to load (different sets have different
             folds structure).

    Returns:
        labels
        id2label
        folds
        parcel_ids
    """
    fname = os.path.join(BASEDIR, 'npy', 'labels_%s.npz' % str(num))
    d = np.load(fname)
    labels = d['labels']
    labels = ma.masked_where(labels == -1, labels)

    folds = d['folds']
    folds = ma.masked_where(folds == -1, folds)

    return labels, d['id2label'], folds, d['parcel_ids']


def _load_dsm_correction(datestr):
    """
    The compute_dsm_correction scripts compute a per-date correction that
    should be subtracted from the DSM to align it with the others
    """
    fname = os.path.join(BASEDIR, 'npy', 'per_date_dsm_correction.npz')
    d = np.load(fname)
    dates = d['dates']
    correction = d['correction']

    return correction[dates.tolist().index(datestr)]


def load_image(datestr, imgtype, autocorrect_dsm=True):
    """
    Loads an image for the given date and type
    Args:
        datestr: Something like 2013_08_21
        imgtype: Either 'rgb' or 'dsm'
        autocorrect_dsm: If true, will apply DSM correction
    Returns:
        array: This is a masked array that contains the data
               - NxMx3 uint8 array with RGB values for 'rgb'
               - NxM float32 array with elevation values for 'dsm'
    """
    assert imgtype in ['rgb', 'dsm']
    fname = os.path.join(BASEDIR, 'rasters', '%s_%s.tif' % (datestr, imgtype))
    assert os.path.exists(fname), 'File does not exist : %s' % fname

    ds = gdal.Open(fname)
    if imgtype == 'rgb':
        # This should be a RGBA image
        arr = ds.ReadAsArray()
        assert len(arr.shape) == 3
        assert arr.shape[0] == 4
        assert arr.dtype == np.uint8
        arr = np.rollaxis(arr, 0, start=3)
        mask = arr[:,:,3] == 0
        mask = np.dstack([mask, mask, mask])
        return ma.masked_array(arr[:,:,:3], mask=mask)
    else:  # dsm
        assert ds.RasterCount == 1
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        arr = band.ReadAsArray()
        arr = ma.masked_where(arr == nodata, arr)
        assert len(arr.shape) == 2
        assert arr.dtype == np.float32

        if autocorrect_dsm:
            return arr - _load_dsm_correction(datestr)

    return arr


