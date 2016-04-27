import unittest
import tempfile
import utils
import numpy as np
import keras.models as models
from keras_utils import (ExtractWindowsLayer, ExtractHist1DLayer, save_model,
    load_model)
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from numpy.testing.utils import assert_allclose


def simple_histnn_cnn_model(img):
    winsize = 5

    scaler = utils.ImageScaler().fit(img)

    binranges = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]

    m = models.Graph()
    m.scaler = scaler
    m.add_input(name='ij', input_shape=(2,))
    m.add_node(ExtractWindowsLayer(winsize, img),
               name='extract_windows',
               input='ij')
    # the cnn part
    m.add_node(Convolution2D(6, 3, 3),
               name='cnn_conv1',
               input='extract_windows')
    m.add_node(Flatten(), name='cnn_out', input='cnn_conv1')
    # the histnn part
    m.add_node(ExtractHist1DLayer(nbins=20, binranges=binranges),
               name='extract_hist',
               input='extract_windows')
    m.add_node(Flatten(), name='histnn_in', input='extract_hist')
    m.add_node(Dense(10), name='histnn_out', input='histnn_in')

    # merge
    m.add_node(Dense(2), name='merge_dense1', inputs=['cnn_out', 'histnn_out'],
               merge_mode='concat')
    m.add_node(Activation('softmax'), name='softmax', input='merge_dense1')
    m.add_output(name='pred', input='softmax')

    m.compile('adam', {'pred':'categorical_crossentropy'})

    #print_layer_shapes(m, input_shapes={'ij':train_ij[:42].shape})
    return m


class TestSaveLoadModel(unittest.TestCase):
    def test_save_load(self):
        img = np.random.rand(50, 100, 3).astype(np.float32)

        m = simple_histnn_cnn_model(img)
        W1 = m.nodes['cnn_conv1'].W.get_value()
        with tempfile.NamedTemporaryFile(suffix='.zip') as f:
            save_model(f.name, m)
            m2 = load_model(f.name, img)

        W2 = m2.nodes['cnn_conv1'].W.get_value()
        assert_allclose(W1, W2)

if __name__ == '__main__':
    unittest.main()
