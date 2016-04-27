import os
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,optimizer=fast_compile,device=gpu0,floatX=float32'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32'
import sys
sys.path.append('..')

import matplotlib
matplotlib.use('agg')

import argparse
import agronn.keras_utils as keras_utils
import skimage
import skimage.io
import agronn.utils as utils
import numpy as np
import numpy.ma as ma
import pylab as pl
import matplotlib.cm as cm

def main(model_fname, image_fname, outdir, xlim, ylim):
    img = skimage.io.imread(image_fname)
    img = skimage.img_as_float(img[:, :]).astype(np.float32)
    assert img.shape[2] == 4
    mask = img[:,:,3] != 0
    img = img[:,:,:3]

    # load_model applies the scaler to the given image
    model = keras_utils.load_model(model_fname, img)

    utils.mkdir_p(outdir)
    print 'Writing results to ', os.path.abspath(outdir)

    if xlim is None:
        xlim = (0, img.shape[1])
    if ylim is None:
        ylim = (0, img.shape[0])

    mask[:ylim[0], :] = False
    mask[ylim[1]:, :] = False
    mask[:, :xlim[0]] = False
    mask[:, xlim[1]:] = False

    if True:
        pl.figure(figsize=(10,10))
        pl.title('ij_mask')
        pl.imshow(mask, cmap=cm.binary, vmin=0, vmax=1)
        pl.colorbar()
        pl.savefig(os.path.join(outdir, 'ij_mask.png'))
        pl.close()

    ij = np.transpose(np.nonzero(mask))
    print ij.shape

    # Classify by chunk
    chunk_size = 8192
    y_pred = np.zeros(ij.shape[0])
    print 'Classifying ', ij.shape[0], ' points'
    for i in xrange(0, ij.shape[0], chunk_size):
        percentage = '%.2f%%' % (100. * i / float(ij.shape[0]))
        print i, '/', ij.shape[0], percentage
        sys.stdout.flush()

        this_chunk_size = chunk_size
        if i + this_chunk_size > ij.shape[0]:
            this_chunk_size = ij.shape[0] - i
        ij_chunk = ij[i:i+this_chunk_size, :]
        _pred = model.predict({'ij':ij_chunk})['output']
        # TODO: This is kind of an ugly way to figure out nclasses, but
        # it works
        nclasses = _pred.shape[1]
        _pred = _pred.argmax(axis=1)
        y_pred[i:i+this_chunk_size] = _pred
    print 'done'

    #y_pred = model.predict({'ij':ij})['output'].argmax(axis=1)
    # Save prediction
    pred_img = ma.masked_all(img.shape[:2])
    pred_img[mask] = y_pred

    np.savez(os.path.join(outdir, 'pred_img.npz'),
             pred_img=pred_img.filled(-1),
             pred_mask=pred_img.mask)

    if True:
        pl.figure(figsize=(10,10))
        pl.title('pred_img')
        pl.imshow(pred_img, cmap=cm.Paired, vmin=0, vmax=nclasses)
        pl.colorbar(shrink=0.7)
        pl.savefig(os.path.join(outdir, 'pred_img.png'))
        pl.close()

    if True:
        pl.figure(figsize=(10,10))
        pl.title('img')
        pl.imshow(img)
        pl.colorbar(shrink=0.7)
        pl.savefig(os.path.join(outdir, 'input_img.png'))
        pl.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a CNN to classify an image")
    parser.add_argument('model_file', help='The .zip file containing the cnn')
    parser.add_argument('image_file', help="The image file to classify")
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('--xlim', nargs=2, default=None, type=int,
                        help="Specify x limits of the bounding rect")
    parser.add_argument('--ylim', nargs=2, default=None, type=int,
                        help="Speficy y limits of the bounding rect")

    args = parser.parse_args()
    main(args.model_file, args.image_file, args.out_dir, args.xlim, args.ylim)
