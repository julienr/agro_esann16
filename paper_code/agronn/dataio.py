# Save to HDF because cPickle fails with very large arrays
# https://github.com/numpy/numpy/issues/2396
import h5py
import numpy as np
import tempfile
import unittest

def dict_to_hdf(fname, d):
    """
    Save a dict-of-dict datastructure where values are numpy arrays
    to a .hdf5 file
    """
    with h5py.File(fname, 'w') as f:
        def _dict_to_group(root, d):
            for key, val in d.iteritems():
                if isinstance(val, dict):
                    grp = root.create_group(key)
                    _dict_to_group(grp, val)
                else:
                    root.create_dataset(key, data=val)

        _dict_to_group(f, d)

def hdf_to_dict(fname):
    """
    Loads a dataset saved using dict_to_hdf
    """
    with h5py.File(fname, 'r') as f:
        def _load_to_dict(root):
            d = {}
            for key, val in root.iteritems():
                if isinstance(val, h5py.Group):
                    d[key] = _load_to_dict(val)
                else:
                    d[key] = val.value
            return d
        return _load_to_dict(f)

def load(exp_name, ret_d=False, data_fname='data.hdf5'):
    d = hdf_to_dict('../%s' % data_fname)
    mosaic = d['mosaic']
    id2label = d['id2label']
    train_ij = d['experiments'][exp_name]['train_ij']
    test_ij = d['experiments'][exp_name]['test_ij']
    y_train = d['experiments'][exp_name]['y_train']
    y_test = d['experiments'][exp_name]['y_test']
    if ret_d:
        return mosaic, id2label, train_ij, test_ij, y_train, y_test, d
    else:
        return mosaic, id2label, train_ij, test_ij, y_train, y_test

# -- Unit tests
class HDFIOTest(unittest.TestCase):
    def test_hdfio(self):
        d = {
            'a' : np.random.rand(5, 3),
            'b' : {
                'c' : np.random.randn(1, 2),
                'd' : {
                    'e' : np.random.randn(10, 5),
                    'f' : np.random.randn(10, 5),
                }
            }
        }
        with tempfile.NamedTemporaryFile() as f:
            dict_to_hdf(f.name, d)
            d2 = hdf_to_dict(f.name)
            self.assertItemsEqual(d, d2)

if __name__ == '__main__':
    unittest.main()
