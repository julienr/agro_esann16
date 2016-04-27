import os
import errno
import cPickle
import numpy as np
import numpy.ma as ma
import pylab as pl
import matplotlib.cm as cm
import skimage.transform as sktransform
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from six import string_types

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST:
            pass
        else: raise

def pickle_save(filename, obj):
    with open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open(filename, 'rb') as f:
        d = cPickle.load(f)
    return d

def norm01(arr):
    amin = arr.min()
    return (arr - amin) / (arr.max() - amin)


class WindowScaler(object):
    """
    Like sklearn.preprocessing.StandardScaler, but works on image windows
    """
    def fit(self, X, std_epsilon=1e-3):
        """
        Args:
            X: (nexamples, nchannels, win_width, win_height)
        """
        # per-channel mean
        self._mean = []
        self._std = []
        for chan in xrange(X.shape[1]):
            self._mean.append(X[:,chan,:,:].mean())
            chan_std = X[:,chan,:,:].std()
            if chan_std < std_epsilon:
                chan_std = std_epsilon
            self._std.append(chan_std)
        self._mean = np.array(self._mean)
        self._std = np.array(self._std)
        return self

    def transform(self, X):
        return (X - self._mean[None, :, None, None]) / self._std[None, :, None, None]

    def inverse_transform(self, X):
        return self._mean[None, :, None, None] + X * self._std[None, :, None, None]

    def __repr__(self):
        return 'WindowScaler(\n  mean=%s\n  var=%s\n)' % (str(self._mean), str(self._std))


class ImageScaler(object):
    """
    Thin wrapper around sklearn.preprocessing.StandardScaler that works on image
    (and maintain their shapes). Doing per-channel scaling/centering
    """
    def fit(self, img):
        """
        Args:
            img: (width, height, nchans)
        """
        self._scaler = StandardScaler().fit(img.reshape(-1, img.shape[2]))
        return self

    def transform(self, img):
        return self._scaler.transform(img.reshape(-1, img.shape[2])).reshape(*img.shape)

    def inverse_transform(self, img):
        return self._scaler.inverse_transform(img.reshape(-1, img.shape[2])).reshape(*img.shape)

    def __repr__(self):
        return 'ImageScaler(\n  %s\n  mean=%s\n  std=%s\n)' % (self._scaler, self._scaler.mean_, self._scaler.std_)


class ImageCenterer(object):
    """
    Assuming a [0,1] image, center it to [-0.5, 0.5]
    """
    def fit(self, img):
        assert img.max() <= 1.0
        return self

    def transform(self, img):
        return img - 0.5

    def inverse_transform(self, img):
        return img + 0.5



# http://skll.readthedocs.org/en/latest/_modules/skll/metrics.html#kappa
def kappa(y_true, y_pred, weights=None, allow_off_by_one=False):
    """
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.

    In the course of calculating kappa, all items in `y_true` and `y_pred` will
    first be converted to floats and then rounded to integers.

    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.

    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :param weights: Specifies the weight matrix for the calculation.
                    Options are:

                        -  None = unweighted-kappa
                        -  'quadratic' = quadratic-weighted kappa
                        -  'linear' = linear-weighted kappa
                        -  two-dimensional numpy array = a custom matrix of
                           weights. Each weight corresponds to the
                           :math:`w_{ij}` values in the wikipedia description
                           of how to calculate weighted Cohen's kappa.

    :type weights: str or numpy array
    :param allow_off_by_one: If true, ratings that are off by one are counted as
                             equal, and all other differences are reduced by
                             one. For example, 1 and 2 will be considered to be
                             equal, whereas 1 and 3 will have a difference of 1
                             for when building the weights matrix.
    :type allow_off_by_one: bool
    """
    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    # Note: numpy and python 3.3 use bankers' rounding.
    try:
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError as e:
        raise RuntimeError("For kappa, the labels should be integers or strings "
                     "that can be converted to ints (E.g., '4.0' or '3').")

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, string_types):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == 'linear':
                    weights[i, j] = diff
                elif wt_scheme == 'quadratic':
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  # unweighted
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError('Invalid weight scheme specified for '
                                     'kappa: {}'.format(wt_scheme))

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k

# utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)

def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    assert nrows * ncols >= nimgs
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


def pad_mosaic(img, padding):
    padded = ma.masked_all((2*padding + img.shape[0], 2*padding + img.shape[1]),
                           dtype=img.dtype)
    padded[padding:-padding, padding:-padding] = img
    return padded
    

def make_mosaic_rgb(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    assert nrows * ncols >= nimgs
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border,
                            3),
                            dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

#l.imshow(make_mosaic(np.random.random((9, 10, 10)), 3, 3, border=1), interpolation='nearest')

def rotation_transform(angle, center):
    tform1 = sktransform.SimilarityTransform(translation=-center)
    tform2 = sktransform.SimilarityTransform(rotation=np.deg2rad(angle))
    tform3 = sktransform.SimilarityTransform(translation=center)
    tform = tform1 + tform2 + tform3
    return tform


class RotatedImageMosaicBuilder(object):
    """
    Given an image and a number of rotation angles, builds a mosaic
    image that contain rotated copies of the original image.
    """

    def __init__(self, img, rot_angles=None):
        if rot_angles is None:
            rot_angles = [0]

        # The image has to be padded to be able to fully include the circumscribed circle of the
        # square defined by max of w/h of original image (so when it is rotated,
        # we don't cut anything)
        longest_side = max(img.shape[0], img.shape[1])
        circum_diameter = np.sqrt(2 * longest_side**2)

        pad_i = int(np.ceil((circum_diameter - img.shape[0]) / 2.0))
        pad_j = int(np.ceil((circum_diameter - img.shape[1]) / 2.0))

        img_padded = np.zeros((img.shape[0] + 2*pad_i,
                               img.shape[1] + 2*pad_j, img.shape[2]),
                                  dtype=img.dtype)
        img_padded[pad_i:img_padded.shape[0]-pad_i,
                   pad_j:img_padded.shape[1]-pad_j, :] = img

        center = np.array((img.shape[1], img.shape[0])) / 2. - 0.5
        padded_center = np.array((img_padded.shape[1], img_padded.shape[0]))
        padded_center = padded_center / 2. - 0.5

        rotated_imgs = []
        ij_transforms = []
        for angle in rot_angles:
            if angle == 0:
                rotated_imgs.append(img_padded)
                ij_transforms.append(sktransform.AffineTransform())
            else:
                transf = rotation_transform(angle, padded_center)
                rotated = sktransform.warp(img_padded, transf,
                        mode='constant', cval=0, order=0)
                # skimage.transform.rotate implicitely converts to float32.
                # So we have to convert back to uint if this is the input type
                if img.dtype == np.uint8:
                    rotated = skimage.img_as_ubyte(rotated)
                rotated_imgs.append(rotated)
                # ij in get_windows are given in original (non-padded)
                # coordinates, so we have to use a non-padded transform as well
                ij_transforms.append(rotation_transform(angle, center))

        self.rot_angles = rot_angles
        self.padded_shape = img_padded.shape
        self.ij_transforms = ij_transforms
        self.pad_i = pad_i
        self.pad_j = pad_j
        self.mosaic = np.concatenate(rotated_imgs, axis=0)

    def get_mosaic(self):
        return self.mosaic

    def get_mosaic_ij(self, ij):
        """
        Given a set of ij coordinates on the original image, computes the
        corresponding point on each image in the mosaic, therefore duplicating
        the points.
        That is what you want to do for your training points.
        If ij is Nx2 and there are R rotations, this returns a RxNx2 array
        """
        out_ijs = []
        for imnum, transf in enumerate(self.ij_transforms):
            tij = transf.inverse(ij[:,::-1]).squeeze()[:,::-1]
            tij = tij.astype(np.int32)

            # images are vertically stacked, so translate ij accordingly
            tij[:,0] += imnum * (self.padded_shape[0]) + self.pad_i
            tij[:,1] += self.pad_j

            out_ijs.append(tij)
        return np.array(out_ijs)

    def get_padded_ij(self, ij):
        """
        Given a set of ij coordinates, simply add pad_i, pad_j to them
        This is what you want to do for the test points
        """
        out_ij = ij.copy()
        out_ij[:, 0] += self.pad_i
        out_ij[:, 1] += self.pad_j
        return out_ij

    def transform_ij_y(self, train_ij, y_train, test_ij, y_test):
        """
        Shortcut function
        """
        _train_ij = self.get_mosaic_ij(train_ij)
        _train_ij = _train_ij.reshape(-1, 2)
        _y_train = np.tile(y_train, len(self.rot_angles))
        _test_ij = self.get_padded_ij(test_ij)

        return _train_ij, _y_train, _test_ij, y_test
