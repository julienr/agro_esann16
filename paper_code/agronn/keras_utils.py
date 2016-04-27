import os
import numpy as np
import zipfile
import tempfile
import my_ops
import theano
from keras.layers.core import Layer
import keras.models
import utils
import json

def print_model_params(model):
    print '-- Parameters'
    nparams = 0
    for W in model.get_weights():
        print W.shape
        nparams += W.size
    print "nparams : ", nparams


def save_model_with_weights(fname, model, scaler, metadata=None):
    """
    Save a keras model config, weights and scaler into a .zip file
    metadata is a dict of arbitrary informations
    """
    if metadata is None:
        metadata = {}

    # We'll bundle the config JSON and the weights HDF5 into a .zip
    with zipfile.ZipFile(fname, mode='w') as zipf:
        # First, save weights to a temporary hdf5 file and then zip it
        with tempfile.NamedTemporaryFile() as wf:
            model.save_weights(wf.name, overwrite=True)
            zipf.write(wf.name, arcname='weights.hdf5')
        with tempfile.NamedTemporaryFile() as wf:
            utils.pickle_save(wf.name, scaler)
            zipf.write(wf.name, arcname='scaler.pickle')
        # add the winsize to the model config
        config = model.get_config()
        config['metadata'] = metadata
        json_str = json.dumps(config)
        zipf.writestr('config.json', json_str)


def load_scaler(fname):
    """
    Load just the scaler from a model .zip
    """
    with zipfile.ZipFile(fname, mode='r') as zipf:
        # scaler - extract to tmpfname
        wf, tmpfname = tempfile.mkstemp()
        with open(tmpfname, 'w') as sf:
            with zipf.open('scaler.pickle') as zsf:
                sf.write(zsf.read())
        scaler = utils.pickle_load(sf.name)
        os.remove(tmpfname)
    return scaler


def load_model_with_weights(fname, custom_layers=None):
    """
    Loads model and scaler from .zip
    """
    if custom_layers is None:
        custom_layers = {}

    with zipfile.ZipFile(fname, mode='r') as zipf:
        # First, instantiate model from config
        json_str = zipf.read('config.json')
        config = json.loads(json_str)
        metadata = config['metadata']
        del config['metadata']
        model = keras.models.model_from_config(config,
            custom_objects=custom_layers)

        # Unzip weights.hdf to a temporary file and load it
        wf, tmpfname = tempfile.mkstemp()
        # weights - extract to tmpfname
        with open(tmpfname, 'w') as wf:
            with zipf.open('weights.hdf5') as zwf:
                wf.write(zwf.read())
        model.load_weights(tmpfname)

        # scaler - extract to tmpfname
        with open(tmpfname, 'w') as sf:
            with zipf.open('scaler.pickle') as zsf:
                sf.write(zsf.read())
        scaler = utils.pickle_load(sf.name)
        os.remove(tmpfname)

    return model, scaler, metadata


class ExtractWindowsLayer(Layer):
    # See Merge layer implementaton for multiple inputs layer
    # https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L303
    def __init__(self, winsize, img, **kwargs):
        """
        Img should be directly supplied as a **numpy** array since it is constant
        across minibatches (we batch win_ij instead). keras layers always have a 'None'
        first dimension that indicates the batch axis
        """
        self.winsize = winsize
        self.op = my_ops.extract_2d_windows(winsize)
        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        self.img_var = theano.shared(img, borrow=True)
        self.img_shape = img.shape

    @property
    def output_shape(self):
        N = self.input_shape[0]
        nchans = self.img_shape[2]
        return (N, nchans, self.winsize, self.winsize)

    def get_output(self, train=False):
        return self.op(self.img_var, self.get_input(train))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "winsize": self.winsize}
        base_config = super(ExtractWindowsLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ExtractHist1DLayer(Layer):
    def __init__(self, nbins, binranges, **kwargs):
        self.nbins = nbins
        self.binranges = binranges
        self.op = my_ops.extract_hist_1d(nbins, binranges)
        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []

    @property
    def output_shape(self):
        nchans = self.input_shape[1]
        return (None, nchans, self.nbins)

    def get_output(self, train=False):
        return self.op(self.get_input(train=False))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "nbins": self.nbins,
                  "binranges": self.binranges}
        base_config = super(ExtractHist1DLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IJLayer(Layer):
    def __init__(self, **kwargs):
        super(IJLayer, self).__init__(**kwargs)
        # The first dimension (batch) is added automatically in set_input_shape
        self.set_input_shape((2,))


def save_model(fname, model, metadata=None):
    assert hasattr(model, 'scaler')
    save_model_with_weights(fname, model, model.scaler, metadata)


def load_model(fname, img):
    # We need the scaler to instantiate the ExtractWindowsLayer below
    scaler = load_scaler(fname)

    # Since ExtractWindowsLayer relies on an external theano variable
    # for the image, we have to customize its instantiation here
    def extract_windows_layer_factory(winsize):
        return ExtractWindowsLayer(winsize, scaler.transform(img))
    custom_layers = {
        'ExtractWindowsLayer' : extract_windows_layer_factory,
        'ExtractHist1DLayer' : ExtractHist1DLayer
    }

    model, _, meta = load_model_with_weights(
        fname, custom_layers=custom_layers)
    model.scaler = scaler
    return model
