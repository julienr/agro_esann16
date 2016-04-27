"""Custom theano operators"""
from __future__ import division
import sys
import theano
import numpy as np
try:
    import theano.sandbox.cuda as cuda
    from pycuda.compiler import SourceModule
    import pycuda
    import pycuda.gpuarray
    # This is required otherwise we'll get "no current device" error
    import theano.misc.pycuda_init
    from theano.sandbox.cuda import gpu_from_host
    has_cuda = True
except ImportError as e:
    has_cuda = False

from theano.gof.opt import local_optimizer


def _windows_extract_infer_shape(hws1, hws2, input_shapes):
    imshp = input_shapes[0]  # 3D image shape
    ijshp = input_shapes[1]  # 2D ij points shape

    nchans = imshp[2]
    nwins = ijshp[0]
    # TODO: This assert doesn't work because those are symbolic theano
    # shapes. Not sure if we can assert here...
    # assert ijshp[1] == 2, "You must provide ij as a Nx2 array, " \
    #        "got %s" % str(ijshp)

    winsize = hws1 + hws2
    outshp = (nwins, nchans, winsize, winsize)
    return [outshp]


def _windows_extract_compute_hws(winsize):
    """This returns the interval [i-hws1:i+hws1] suitable for indexing"""
    if winsize % 2 == 1:
        hws1 = (winsize - 1) // 2
        hws2 = hws1 + 1
    else:
        hws1 = winsize // 2
        hws2 = hws1

    assert hws1 + hws2 == winsize

    return hws1, hws2


class Extract2DWindowsOp(theano.Op):
    """
    A theano operator that extract 2D windows of fixed size around
    given input points.
    The input should be :
    - The image (height * width * nchannels) from which to extract windows
    - a Nx2 vector of int giving the pixel locations of the center of the
      windows.
    The output will be a N x nchannels x winsize x winsize array of all
    the windows, suitable for use in a CNN for example.
    """
    # A window centered at ij corresponds to img[i-hws1:i+hws2, j-hws1:j+hws2].
    # For even window, hws1 = hws2, for odd windows, one will have one pixel
    # more than the other
    __props__ = ('hws1', 'hws2')

    def __init__(self, hws1, hws2):
        super(Extract2DWindowsOp, self).__init__()
        self.hws1 = hws1
        self.hws2 = hws2

    def make_node(self, img, win_ij):
        assert hasattr(self, '_props'), "Your version of theano is too old " \
            "to support __props__."
        img = theano.tensor.as_tensor_variable(img)
        win_ij = theano.tensor.as_tensor_variable(win_ij.astype('int32'))

        output = theano.tensor.tensor4(dtype=img.type.dtype)

        return theano.Apply(self, [img, win_ij], [output])

    def infer_shape(self, node, input_shapes):
        return _windows_extract_infer_shape(self.hws1, self.hws2, input_shapes)

    def perform(self, node, inputs, output_storage):
        img = inputs[0]
        imshp = img.shape
        win_ij = inputs[1]
        # a list of storage cell (each cell is a one-element list to pass by
        # reference)
        z = output_storage[0]
        if z[0] is None:
            # Need to allocate
            winsize = self.hws1 + self.hws2
            nwins = win_ij.shape[0]
            nchans = img.shape[2]
            z[0] = np.zeros((nwins, nchans, winsize, winsize), dtype=img.dtype)
        zz = z[0]

        for cnt, ij in enumerate(win_ij):
            i, j = ij
            #assert i >= self.hws1, 'i : %d, self.hws1 %d' % (i, self.hws1)
            #assert i <= imshp[0] - self.hws2
            #assert j >= self.hws1
            #assert j <= imshp[1] - self.hws2
            # For some reason, keras first call this with win_ij full of
            # zeros, which would cause the asserts to fail. Instead,
            # skip windows that are incomplete as this seems to work
            # regardless
            inside = (i >= self.hws1) and\
                     (i <= imshp[0] - self.hws2) and\
                     (j >= self.hws1) and\
                     (j <= imshp[1] - self.hws2)

            if not inside:
                continue
            zz[cnt, :, :, :] = \
                img[i-self.hws1:i+self.hws2,
                    j-self.hws1:j+self.hws2, :].transpose(2, 0, 1)

        z[0] = zz


if has_cuda:
    class Extract2DWindowsGpuOp(theano.sandbox.cuda.GpuOp):
        """Like Extract2DWindows, but runs on the GPU"""
        __props__ = ('hws1', 'hws2')

        def __init__(self, hws1, hws2):
            """See Extract2DWindowsOp.__init__"""
            super(Extract2DWindowsGpuOp, self).__init__()
            self.hws1 = hws1
            self.hws2 = hws2

        def make_node(self, img, win_ij):
            assert hasattr(self, '_props'), "Your version of theano is too old " \
                "to support __props__."
            # Theano's CudaNdArray support strides. But this require writing C
            # code calling the functions of sandbox/cuda/cuda_ndarray.cuh
            # and passing all the strides to the kernel to do the correct
            # computation. Instead, enforce contiguous arrays.
            cu_img = cuda.basic_ops.gpu_contiguous(
                cuda.basic_ops.as_cuda_ndarray_variable(img))
            assert cu_img.dtype == 'float32'
            # CudaNdArray only supports float32, so cast to float
            cu_win_ij = cuda.basic_ops.gpu_contiguous(
                cuda.basic_ops.as_cuda_ndarray_variable(win_ij.astype('float32')))
            assert cu_win_ij.dtype == 'float32'

            output = cuda.CudaNdarrayType(
                dtype=img.type.dtype,
                broadcastable=[False, False, False, False])()
            return theano.Apply(self, [cu_img, cu_win_ij], [output])

        def infer_shape(self, node, input_shapes):
            return _windows_extract_infer_shape(self.hws1, self.hws2, input_shapes)

        def make_thunk(self, node, storage_map, compute_map, no_recycling):
            mod = SourceModule(
            """
            // We are given an input image of size (imwidth, imheight, nchans)
            // and we should extract windows at given positions (win_ij) and
            // return a (N, nchans, winsize, winsize) array containing the
            // extracted windows
            __global__ void extract_2d_windows(
                float* img, float* win_ij, float* out_win, int N, int imwidth,
                int nchans, int hws1, int hws2
            ) {
                int id = blockIdx.x * blockDim.x + threadIdx.x;
                // TODO: Could precompute those
                int winsize = hws1 + hws2;
                int win_offset = id * winsize * winsize * nchans;
                int win_c_stride = winsize * winsize;
                int img_y_stride = imwidth * nchans;

                // It might happen (if N is not a number of the size of a thread
                // block) that we have more thread than needed
                if (id < N) {
                    // Theano only supports float, so we upload as float and cast
                    int i = (int)win_ij[2*id];
                    int j = (int)win_ij[2*id + 1];

                    for (int wi = 0; wi < winsize; ++wi) {
                        for (int wj = 0; wj < winsize; ++wj) {
                            for (int c = 0; c < nchans; ++c) {
                                int img_idx = (i - hws1 + wi) * img_y_stride
                                            + (j - hws1 + wj) * nchans
                                            + c;
                                // TODO: Bounds check (if we are too close to the
                                // image border)

                                // The output shape is
                                // N x nchans x winsize x winsize
                                int win_idx = win_offset
                                            + c * win_c_stride
                                            + wi * winsize
                                            + wj;
                                out_win[win_idx] = img[img_idx];
                            }
                        }
                    }
                }
            }
            """)  # noqa
            pycuda_fct = mod.get_function("extract_2d_windows")
            inputs = [storage_map[v] for v in node.inputs]
            outputs = [storage_map[v] for v in node.outputs]

            # inputs is a dict of cell, where a cell is a single-element list
            # (basically a pointer)
            winsize = self.hws1 + self.hws2

            def thunk():
                imh, imw, nchans = inputs[0][0].shape
                nwins = inputs[1][0].shape[0]

                # print "ij ", np.array(inputs[1][0])

                z = outputs[0]
                if z[0] is None:
                    # Need to allocate
                    z[0] = cuda.CudaNdarray.zeros(
                        (nwins, nchans, winsize, winsize))

                # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy
                # https://github.com/inducer/pycuda/blob/master/pycuda/gpuarray.py#L81
                grid, thread_blocks = pycuda.gpuarray.splay(nwins)
                # print 'grid : ', grid
                # print 'blocks : ', thread_blocks
                pycuda_fct(inputs[0][0], inputs[1][0], z[0], np.intc(nwins),
                           np.intc(imw), np.intc(nchans), np.intc(self.hws1),
                           np.intc(self.hws2),
                           block=thread_blocks, grid=grid)
            thunk.inputs = inputs
            thunk.outputs = outputs
            thunk.lazy = False
            return thunk


class ExtractHist1DOp(theano.Op):
    """
    A theano operator that extract 1d per-channel histograms from an image.
    The input should be :
    - The images (N * nchannels * height * width) from which to extract windows

    The output will be a N x nchannels x nbins array of histograms
    """
    __props__ = ('nbins', 'binranges')

    def __init__(self, nbins, binranges):
        super(ExtractHist1DOp, self).__init__()
        self.nbins = nbins
        # Theano props cannot be lists or it will fail with "unhashable type"
        # This is important when deserializing from JSON because JSON will serialize tuple
        # as lists
        self.binranges = tuple(tuple(br) for br in binranges)

    def make_node(self, img):
        assert hasattr(self, '_props'), "Your version of theano is too old " \
            "to support __props__."
        img = theano.tensor.as_tensor_variable(img)

        # N x nchannels x nbins
        output = theano.tensor.tensor3(dtype=theano.config.floatX)

        return theano.Apply(self, [img], [output])

    def infer_shape(self, node, input_shapes):
        imshp = input_shapes[0]  # 4D windows shape
        batch_size, nchans = imshp[0], imshp[1]
        outshp = (batch_size, nchans, self.nbins)
        return [outshp]

    def perform(self, node, inputs, output_storage):
        img = inputs[0]
        imshp = img.shape
        # a list of storage cell (each cell is a one-element list to pass by
        # reference)
        z = output_storage[0]
        if z[0] is None:
            # Need to allocate
            batch_size, nchans = imshp[0], imshp[1]
            z[0] = np.zeros((batch_size, nchans, self.nbins),
                            dtype=np.float32)

        zz = z[0]

        npixels = imshp[2] * imshp[3]
        for bi in xrange(imshp[0]):
            for c in xrange(imshp[1]):
                h1d = np.histogram(img[bi, c], bins=self.nbins,
                                   range=self.binranges[c])[0]
                h1d = h1d.astype(np.float32) / npixels
                zz[bi, c, :] = h1d

        z[0] = zz


if has_cuda:
    class ExtractHist1DGpuOp(theano.sandbox.cuda.GpuOp):
        """Like Extract2DWindowsOp, but for the GPU"""
        __props__ = ('nbins', 'binranges')

        def __init__(self, nbins, binranges):
            super(ExtractHist1DGpuOp, self).__init__()
            self.nbins = nbins
            # Theano props cannot be lists or it will fail with "unhashable type"
            # This is important when deserializing from JSON because JSON will serialize tuple
            # as lists
            self.binranges = tuple(tuple(br) for br in binranges)

        def make_node(self, img):
            assert hasattr(self, '_props'), "Your version of theano is too old " \
                "to support __props__."
            # Theano's CudaNdArray support strides. But this require writing C
            # code calling the functions of sandbox/cuda/cuda_ndarray.cuh
            # and passing all the strides to the kernel to do the correct
            # computation. Instead, enforce contiguous arrays.
            cu_img = cuda.basic_ops.gpu_contiguous(
                cuda.basic_ops.as_cuda_ndarray_variable(img))
            assert cu_img.dtype == 'float32'

            # N x nchannels x nbins
            output = cuda.CudaNdarrayType(
                dtype='float32',
                broadcastable=[False, False, False])()
            return theano.Apply(self, [cu_img], [output])

        def infer_shape(self, node, input_shapes):
            imshp = input_shapes[0]  # 4D windows shape
            batch_size, nchans, imw, imh = imshp
            outshp = (batch_size, nchans, self.nbins)
            return [outshp]

        def make_thunk(self, node, storage_map, compute_map, no_recycling):
            mod = SourceModule(
            """
            // We are given a batch of window of size
            // (N, nchans, winsize, winsize) and we should output a corresponding
            // batch of per-channel histograms of size
            // (N, nchans, nbins)
            //
            // The histograms are parametrized by the number of bins and, for
            // each channel, the range of values (x0, x1).
            // To compute the bin a value will fall into, we use
            //   bin = (val - x0) / ((x1 - x0) / nbins)
            //       = nbins * (val - x0) / (x1 - x0)
            //       = (val - x0) * nbins / (x1 - x0)
            //       = (val - x0) * cx
            // with cx = nbins / (x1 - x0)
            __global__ void compute_hist1d(
                float* win_batch, float* out_hists, int N, int nchans,
                int winw, int winh, int nbins, float* x0, float* x1, float* cx
            ) {
                int id = blockIdx.x * blockDim.x + threadIdx.x;

                // TODO: Could precompute those
                int win_stride = winw * winh * nchans;
                int hist_stride = nchans * nbins;
                float norm_factor = 1.0f / (winw * winh);

                // It might happen (if N is not a number of the size of a thread
                // block) that we have more thread than needed
                if (id < N) {
                    // initialize hists to 0
                    {
                        int h_offset = id * hist_stride;
                        for (int i = 0; i < nchans * nbins; ++i) {
                            out_hists[h_offset + i] = 0;
                        }
                    }

                    for (int c = 0; c < nchans; ++c) {
                        float* Wc = &win_batch[id * win_stride +
                                               c * winw * winh];
                        float* H = &out_hists[id * hist_stride + c * nbins];

                        for (int wi = 0; wi < winh; ++wi) {
                            for (int wj = 0; wj < winw; ++wj) {
                                float val = Wc[wi * winw + wj];
                                float fbin = (val - x0[c]) * cx[c];
                                // Like np.histogram, for the rightmost bin, we want
                                // values equal to the right edge to be counted in
                                // the last bin, not as outliers
                                // Do NOT use an fabs here as this creates errors
                                // for negative numbers
                                int ix;
                                if (fbin >= (float)nbins && (val - x1[c]) < 1e-6) {
                                    ix = nbins - 1;
                                } else {
                                    // floor is necessary to correctly handle
                                    // negative numbers
                                    ix = (int) floor(fbin);
                                }

                                if (ix >= 0 and ix < nbins) {
                                    H[ix] += 1;
                                }
                            }
                        }
                        // Normalize histograms
                        for (int b = 0; b < nbins; ++b) {
                            H[b] *= norm_factor;
                        }
                    }
                }
            }
            """)  # noqa
            pycuda_fct = mod.get_function("compute_hist1d")
            inputs = [storage_map[v] for v in node.inputs]
            outputs = [storage_map[v] for v in node.outputs]
            # inputs is a dict of cell, where a cell is a single-element list
            # (basically a pointer)

            def thunk():
                N, nchans, winw, winh = inputs[0][0].shape
                nbins = self.nbins
                x0 = np.array([br[0] for br in self.binranges], dtype=np.float32)
                x0 = cuda.CudaNdarray(x0)
                x1 = np.array([br[1] for br in self.binranges], dtype=np.float32)
                x1 = cuda.CudaNdarray(x1)
                cx = np.array([nbins / (br[1] - br[0]) for br in self.binranges],
                              dtype=np.float32)
                cx = cuda.CudaNdarray(cx)

                z = outputs[0]
                if z[0] is None:
                    # Need to allocate
                    z[0] = cuda.CudaNdarray.zeros((N, nchans, nbins))

                # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy
                # https://github.com/inducer/pycuda/blob/master/pycuda/gpuarray.py#L81
                grid, thread_blocks = pycuda.gpuarray.splay(N)
                pycuda_fct(inputs[0][0], z[0], np.intc(N), np.intc(nchans),
                           np.intc(winw), np.intc(winh), np.intc(nbins),
                           x0, x1, cx,
                           block=thread_blocks, grid=grid)
            thunk.inputs = inputs
            thunk.outputs = outputs
            thunk.lazy = False
            return thunk


def extract_2d_windows(winsize):
    hws1, hws2 = _windows_extract_compute_hws(winsize)
    return Extract2DWindowsOp(hws1, hws2)


def extract_hist_1d(nbins, binranges):
    return ExtractHist1DOp(nbins, binranges)


if has_cuda:
    def register_cuda_op_substitute(CpuOpCls, GpuOpCls):
        """
        A helper function that creates a local optimizer that will substitues
        the given CPU operator by the given GPU one.
        This is heavily inspired by the way theano.sandbox.cuda.opt.local_gpu_conv
        optimizer works :

        https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/opt.py#L1432
        """
        @local_optimizer([cuda.gpu_from_host, CpuOpCls])
        def optimize(node):
            if isinstance(node.op, cuda.GpuFromHost):
                # gpu_from_host(cpu_op) -> gpu_op(gpu_from_host)
                host_input = node.inputs[0]

                if host_input.owner and isinstance(host_input.owner.op, CpuOpCls):
                    cpu_op = host_input.owner.op
                    args = dict(zip(cpu_op.__props__, cpu_op._props()))
                    gpu_op = GpuOpCls(**args)
                    inputs = host_input.owner.inputs
                    out = gpu_op(*inputs)
                    return [out]

            if isinstance(node.op, CpuOpCls):
                # cpu_op(host_from_gpu) -> host_from_gpu(gpu_op)
                def _is_variable_on_gpu(var):
                    return var.owner and isinstance(var.owner.op, cuda.HostFromGpu)
                inputs = node.inputs
                inputs_on_gpu = map(_is_variable_on_gpu, inputs)

                if any(inputs_on_gpu):
                    cpu_op = node.op
                    args = dict(zip(cpu_op.__props__, cpu_op._props()))
                    gpu_op = GpuOpCls(**args)
                    out = gpu_op(*inputs)
                    out = cuda.host_from_gpu(out)
                    return [out]

            return False

        optname = '%s_to_gpu' % CpuOpCls.__name__
        if optname not in cuda.gpu_optimizer:
            # print 'registering ', optname
            cuda.gpu_optimizer.register(
                optname, optimize, 'fast_run', 'fast_compile', 'gpu'
            )

    try:
        reloading
    except NameError:
        reloading = False
    else:
        reloading = True
        # reload() breaks instanceof (because it redefines class). This cause our
        # optimizer to fail (because they are registered once, on the old classes)
        import warnings
        warnings.warn('reload() breaks automatic GPU optimization on this module')

    register_cuda_op_substitute(Extract2DWindowsOp, Extract2DWindowsGpuOp)
    register_cuda_op_substitute(ExtractHist1DOp, ExtractHist1DGpuOp)
