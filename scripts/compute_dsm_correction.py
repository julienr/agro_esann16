"""
This computes a vertical correction for each DSM based on some ground control
points that are manually placed on points that are invariant
across time (e.g. roads)

This will output plots in a _log directory
"""
##
import os
import random
import numpy as np
import numpy.ma as ma
import pylab as pl
import matplotlib.cm as cm
import skimage.color as skcolor
from osgeo import gdal
from osgeo import ogr
import pandas as pd
import gdal
from osgeo import ogr, osr
##

def map_to_pixel_coords(mx, my, geot):
    #Convert from map to pixel coordinates.
    #Only works for geotransforms with no rotation.
    #If raster is rotated, see http://code.google.com/p/metageta/source/browse/trunk/metageta/geometry.py#493
    # no rotation
    assert geot[2] == 0
    assert geot[4] == 0

    px = int((mx - geot[0]) / geot[1]) #x pixel
    py = int((my - geot[3]) / geot[5]) #y pixel

    return px, py

def load_rgb(date):
    # We have to use the unclipped DSM otherwise we don't have roads
    fname = os.path.join(os.environ['AGRODATA'], '2_images', '5cm',
                         'agroscope_%s_rgb_5cm.tif' % date)
    ds = gdal.Open(fname)
    arr = ds.ReadAsArray()
    arr = np.rollaxis(arr, 0, start=3)
    mask = arr[:,:,3] == 0
    mask = np.dstack([mask, mask, mask])
    arr = ma.masked_array(arr[:,:,:3], mask=mask)

    sr = osr.SpatialReference(wkt=ds.GetProjection())
    geot = ds.GetGeoTransform()

    return arr, sr, geot

def load_dsm(date):
    # We have to use the unclipped DSM otherwise we don't have roads
    fname = os.path.join(os.environ['AGRODATA'], '2_images', '5cm',
                         'agroscope_%s_dsm_5cm.tif' % date)
    ds = gdal.Open(fname)
    assert ds is not None
    assert ds.RasterCount == 1
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr = band.ReadAsArray()
    arr = ma.masked_where(arr == nodata, arr)

    dsm_sr = osr.SpatialReference(wkt=ds.GetProjection())
    geot = ds.GetGeoTransform()

    return arr, dsm_sr, geot

def load_ground_points(img_sr, img_geot):
    driver = ogr.GetDriverByName('Geojson')

    fname = os.path.join('shapefiles', 'dsm_groundpoints.geojson')
    ds = driver.Open(fname)
    assert ds is not None

    points_layer = ds.GetLayer()
    points_layer.ResetReading()
    feature = points_layer.GetNextFeature()

    points = []
    while True:
        feature = points_layer.GetNextFeature()
        if feature is None:
            break
        assert img_sr.IsSame(points_layer.GetSpatialRef())
        #transform = osr.CoordinateTransformation(points_layer.GetSpatialRef(), dsm_sr)
        geom = feature.GetGeometryRef()
        #geom.Transform(transform)
        x, y = map_to_pixel_coords(geom.GetX(), geom.GetY(), img_geot)
        points.append((x, y))
    return points

##
rgb, img_sr, img_geot = load_rgb('2013_10_08')
points = np.array(load_ground_points(img_sr, img_geot))

## Save a plot of ground control points
if True:
    pl.figure(figsize=(15, 15))
    pl.title('Ground control points')
    pl.imshow(rgb)
    pl.scatter(points[:,0], points[:,1], c='r', linewidth=0)
    pl.savefig('_log/control_points.png', dpi=150)
##
dates = ['2013_08_12', '2013_08_21', '2013_08_26', '2013_09_13', '2013_10_08']

# For each point, the height at each date
dates_heights = []

for date in dates:
    dsm, dsm_sr, dsm_geot = load_dsm(date)
    assert dsm_geot == img_geot
    assert dsm_sr.IsSame(img_sr)

    heights = dsm[points[:,1], points[:,0]].filled(-1)
    assert np.all(heights != -1)

    dates_heights.append(heights)
dates_heights = np.array(dates_heights)
##
if False:
    pl.figure(figsize=(15, 15))
    pl.title('GCP height across dates')
    pl.plot(dates_heights)
    pl.xticks(np.arange(len(dates)), dates)
    pl.xlabel('date')
    pl.ylabel('height')
##
# Use date 0 as baseline
baseline = dates_heights[0,:]

# for each date and point, computes the correction 
correction = dates_heights - np.tile(baseline, (len(dates), 1))

# for each date, compute average correction
avg_corr = correction.mean(axis=1)
corr_std = correction.std(axis=1)
print 'Average correction for each date: ', avg_corr
print 'stddev : ', corr_std
##
corrected_heights = dates_heights - avg_corr.reshape(-1, 1)
##
if True:
    pl.figure(figsize=(15, 5))
    ax = pl.subplot(121)
    pl.title('GCP height across dates')
    pl.plot(dates_heights)
    pl.xticks(np.arange(len(dates)), dates, rotation=45)
    pl.xlabel('date')
    pl.ylabel('height')

    pl.subplot(122, sharex=ax, sharey=ax)
    pl.title('GCP corrected height across dates')
    pl.plot(corrected_heights)
    pl.xticks(np.arange(len(dates)), dates, rotation=45)
    pl.xlabel('date')
    pl.ylabel('height')

    pl.tight_layout()

    pl.savefig('_log/corrected_gcp_height.png', dpi=150)
##
# On the following figure, we see that the correction helps a bit, but
# there is still some difference that is not accounted for. This might be
# due to a small tilt in the DSM, which is harder to correct
if True:
    pl.figure(figsize=(15, 5))
    ax = pl.subplot(121)
    pl.title('GCP diff to date 0 across dates')
    pl.plot(dates_heights - baseline)
    pl.xticks(np.arange(len(dates)), dates, rotation=45)
    pl.xlabel('date')
    pl.ylabel('height')

    pl.subplot(122, sharex=ax, sharey=ax)
    pl.title('GCP corrected diff to date 0 across dates')
    pl.plot(corrected_heights - baseline)
    pl.xticks(np.arange(len(dates)), dates, rotation=45)
    pl.xlabel('date')
    pl.ylabel('height')

    pl.tight_layout()

    pl.savefig('_log/corrected_gcp_height_diff_to_date0.png', dpi=150)
##
# per_date_correction should be SUBTRACTED to the DSM value
per_date_correction = avg_corr
np.savez('npy/per_date_dsm_correction.npz',
         correction=per_date_correction,
         dates=dates)
##


