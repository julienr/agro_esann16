import unittest
import numpy as np
from utils import RotatedImageMosaicBuilder
from numpy.testing import assert_array_equal, assert_allclose

class TestMosaicBuilder(unittest.TestCase):
    def test_mosaic(self):
        img = np.random.rand(25, 35, 3)

        # rotate by 90 so y=x, x=y
        mb = RotatedImageMosaicBuilder(img, [0, 90])
        train_ij = np.array([[13, 12],
                             [22, 31]], dtype=np.int)
        y_train = [0, 1]
        test_ij = np.array([[8, 15],
                            [17, 9]], dtype=np.int)
        y_test = [1, 0]

        _train_ij, _y_train, _test_ij, _y_test = mb.transform_ij_y(
            train_ij, y_train, test_ij, y_test)

        padding = np.array([mb.pad_i, mb.pad_j])
        assert_array_equal(_y_train, [0, 1, 0, 1])
        assert_array_equal(y_test, [1, 0])

        def _pixels(img, ij):
            return img[ij[:,0], ij[:,1]]

        # The rotation shouldn't change pixel values
        assert_allclose(_pixels(mb.mosaic, _test_ij),
                        _pixels(img, test_ij))
        assert_allclose(_pixels(mb.mosaic, _train_ij[:2]),
                        _pixels(img, train_ij))
        assert_allclose(_pixels(mb.mosaic, _train_ij[2:]),
                        _pixels(img, train_ij))


if __name__ == '__main__':
    unittest.main()
