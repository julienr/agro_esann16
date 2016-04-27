"""
Run with
    THEANO_FLAGS="device=cpu" theano-nose -v my_ops_test.py
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import theano
import unittest
from theano import config
from my_ops import Extract2DWindowsOp, Extract2DWindowsGpuOp, \
    _windows_extract_compute_hws, extract_2d_windows
from my_ops import ExtractHist1DOp, ExtractHist1DGpuOp, extract_hist_1d

# From
# https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/
#   tests/test_basic_ops.py#L26
if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN')\
        .excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


class TestExtract2DWindows(unittest.TestCase):
    def _validate_2d_windows(self, ashape, win_ij, hws):
        hws1, hws2 = hws
        assert abs(hws1 - hws2) <= 1
        winsize = hws1 + hws2
        op = extract_2d_windows(winsize)

        img_t = theano.tensor.tensor3()
        win_ij_t = theano.tensor.matrix(dtype='int32')
        f = theano.function([img_t, win_ij_t], op(img_t, win_ij_t),
                            mode=mode_without_gpu)
        f2 = theano.function([img_t, win_ij_t], op(img_t, win_ij_t),
                             mode=mode_with_gpu)

        # Check that one is using the CPU version and the other the GPU
        assert Extract2DWindowsOp in \
            [x.op.__class__ for x in f.maker.fgraph.toposort()], \
            "f is not using CPU"

        assert Extract2DWindowsGpuOp in \
            [x.op.__class__ for x in f2.maker.fgraph.toposort()], \
            "f2 is not using GPU (cuda_op_substitute failed ?)"

        img = np.arange(np.prod(ashape)).reshape(*ashape)
        img = img.astype(config.floatX)
        out = f(img, win_ij)
        out = np.array(out)

        out2 = f(img, win_ij)
        out2 = np.array(out2)

        # print '--win_ij\n', win_ij
        # print '--img\n', img.squeeze()
        # print '--out\n', out.squeeze()

        assert_array_equal(out, out2)

        assert_array_equal(
            out.shape,
            (len(win_ij), ashape[2], winsize, winsize)
        )

        for pixi, (i, j) in enumerate(win_ij):
            assert_array_equal(
                out[pixi],
                img[i-hws1:i+hws2, j-hws1:j+hws2].transpose(2, 0, 1)
            )

    def test_extract_windows(self):
        # winsize = 2
        self._validate_2d_windows(
            ashape=(6, 5, 1), win_ij=[(1, 1), (3, 2)], hws=(1, 1))
        # winsize = 3
        self._validate_2d_windows(
            ashape=(6, 5, 1), win_ij=[(1, 1), (3, 2)], hws=(1, 2))

        # winsize = 5
        self._validate_2d_windows(
            ashape=(15, 25, 3), win_ij=[(5, 8), (7, 20)], hws=(2, 3))
        # winsize = 6
        self._validate_2d_windows(
            ashape=(15, 25, 3), win_ij=[(5, 8), (7, 20)], hws=(3, 3))

    def test_compute_hws(self):
        self.assertEqual(_windows_extract_compute_hws(5), (2, 3))
        self.assertEqual(_windows_extract_compute_hws(6), (3, 3))


class TestExtractHist1D(unittest.TestCase):
    def _validate_hist1d(self, img, expected_hist, nbins, binranges):
        op = extract_hist_1d(nbins, binranges)

        img_t = theano.tensor.tensor4()
        f = theano.function([img_t], op(img_t), mode=mode_without_gpu)
        f2 = theano.function([img_t], op(img_t), mode=mode_with_gpu)

        # Check that one is using the CPU version and the other the GPU
        assert ExtractHist1DOp in \
            [x.op.__class__ for x in f.maker.fgraph.toposort()], \
            "f is not using CPU"

        assert ExtractHist1DGpuOp in \
            [x.op.__class__ for x in f2.maker.fgraph.toposort()], \
            "f2 is not using GPU (cuda_op_substitute failed ?)"

        img = img.astype(config.floatX)
        out = f(img)
        out = np.array(out)

        out2 = f2(img)
        out2 = np.array(out2)

        assert_allclose(out, out2, 1e-5)
        assert_allclose(out, expected_hist, 1e-5)

    def test_hist1d_single_channel(self):
        img = np.array([
            # 2 (5), 4 (2), 3 (2), 5(1)
            [[2, 2, 2, 2, 4],
             [4, 3, 2, 3, 5]],
            # 1(2), 9(2), 8(1), 3(5)
            [[1, 1, 9, 9, 8],
             [3, 3, 3, 3, 3]],
            ], dtype=np.float32)

        img = img[:, None, :, :]

        expected_out = np.array([
            [[0, 0, 0.5, 0.2, 0.2, 0.1, 0, 0, 0, 0]],
            [[0, 0.2, 0, 0.5, 0, 0, 0, 0, 0.1, 0.2]]], dtype=np.float32)

        nbins = 10
        binranges = ((0, 10), )

        self._validate_hist1d(img, expected_out, nbins, binranges)

    def test_hist1d_multi_channels(self):
        # 5 RGB images of size 10x10
        imshape = (5, 3, 10, 10)
        img = np.random.rand(*imshape)

        nbins = 5
        binranges = ((0, 1), (0, 0.5), (0.2, 0.8))

        npixels = imshape[2] * imshape[3]
        expected_out = []
        for imi in xrange(img.shape[0]):
            cout = []
            for c in xrange(img.shape[1]):
                cout.append(np.histogram(img[imi, c], bins=nbins,
                                         range=binranges[c])[0])
            expected_out.append(cout)
        expected_out = np.array(expected_out, dtype=np.float32)
        expected_out /= npixels

        self._validate_hist1d(img, expected_out, nbins, binranges)


if __name__ == '__main__':
    unittest.main()
