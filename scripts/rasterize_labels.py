"""
This creates npy/labels.npz which contains a set of arrays corresponding
to rasterized version of shapefiles/labels_fold.geojson
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
##
def load_rgb_tif(filename):
    rgb_dataset = gdal.Open(filename)
    img_rgb = rgb_dataset.ReadAsArray()
    img_rgb= np.rollaxis(img_rgb, 0, start=3)
    mask = img_rgb[:,:,3] == 0
    mask = np.dstack([mask, mask, mask])
    img_rgb = ma.array(img_rgb[:,:,:3], mask=mask)
    # convert it to float because it's easier for us after
    img_rgb = ma.array(img_rgb.astype(np.float32) / 255.0, mask=mask)
    return img_rgb

def rasterize_like(shape_layer, model_dataset, dtype, options, nodata_val=0):
    """
    Given a shapefile, rasterizes it so it has
    the exact same extent as the given model_raster

    `dtype` is a gdal type like gdal.GDT_Byte
    `options` should be a list that will be passed to GDALRasterizeLayers papszOptions, like
        ["ATTRIBUTE=vegetation"]
    """
    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create(
        '',
        model_dataset.RasterXSize,
        model_dataset.RasterYSize,
        1,
        dtype
    )
    mem_raster.SetProjection(model_dataset.GetProjection())
    mem_raster.SetGeoTransform(model_dataset.GetGeoTransform())
    mem_band = mem_raster.GetRasterBand(1)
    mem_band.Fill(nodata_val)
    mem_band.SetNoDataValue(nodata_val)

    # http://gdal.org/gdal__alg_8h.html#adfe5e5d287d6c184aab03acbfa567cb1
    # http://gis.stackexchange.com/questions/31568/gdal-rasterizelayer-doesnt-burn-all-polygons-to-raster
    err = gdal.RasterizeLayer(
        mem_raster,
        [1],
        shape_layer,
        None,
        None,
        [1],
        options
    )
    assert err == gdal.CE_None
    return mem_raster.ReadAsArray()


def collect_labels(layer, label_attrname):
    layer.ResetReading()
    unique_labels = set()
    while True:
        feature = layer.GetNextFeature()
        if feature is None:
            break
        name = feature.GetField(label_attrname)
        name = unicode(name, 'utf8')
        unique_labels.add(name)

    # assign a unique id to each
    id2label = []
    for name in sorted(unique_labels):
        id2label.append(name)
    return id2label

## - Load model raster
fname = 'rasters/2013_08_12_rgb.tif'
img_rgb = load_rgb_tif(fname)
model_ds = gdal.Open(fname)
## - Load labels geojson
# There are 3 attributes : id, fold, name
label_fnames = ['shapefiles/labels_1.geojson',
                'shapefiles/labels_2.geojson',
                'shapefiles/labels_3.geojson',
                'shapefiles/labels_4.geojson',
                'shapefiles/labels_5.geojson',
                'shapefiles/labels_6.geojson',
                'shapefiles/labels_full.geojson']

for label_fname in label_fnames:
    #label_fname = 'shapefiles/labels_%d.geojson' % label_count
    assert os.path.exists(label_fname), "Couldn't open %s" % label_fname
    fields = ['id', 'name', 'fold']

    label_ds = ogr.Open(label_fname)
    label_layer = label_ds.GetLayer()

    id2label = collect_labels(label_layer, 'name')

    # Rasterize parcels id
    parcel_ids = rasterize_like(label_layer, model_ds, gdal.GDT_Int16,
                                ['ALL_TOUCHED', 'ATTRIBUTE=id'], nodata_val=-1)
    # Rasterize folds
    folds = rasterize_like(label_layer, model_ds, gdal.GDT_Int16,
                           ['ALL_TOUCHED', 'ATTRIBUTE=fold'], nodata_val=-1)
    # Rasterize labels. We use attribute filters to rasterize all the polygons
    # with a given label, assign an id to id and move on to the next label
    labels = ma.masked_all((model_ds.RasterYSize, model_ds.RasterXSize),
                           dtype=np.int16)
    for lid, label in enumerate(id2label):
        label_layer.SetAttributeFilter("name='%s'" % label.encode('utf8'))
        limg = rasterize_like(label_layer, model_ds, gdal.GDT_Int16,
                              ['ALL_TOUCHED'])
        labels[limg == 1] = lid
    ##
    # Save to npy
    fname = os.path.splitext(os.path.basename(label_fname))[0]
    out_fname = 'npy/%s.npz' % fname
    np.savez(out_fname,
             parcel_ids=parcel_ids,
             folds=folds,
             labels=ma.filled(labels, -1),
             id2label=np.array(id2label))

    print 'Saved to ', out_fname
## Rasterize labels_full, which 
