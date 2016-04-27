
# coding: utf-8

# # Train CNN vs HistNN vs CNN + HistNN

# This is the main experiment script, which trains a CNN, a HistNN and a merged CNN/HistNN network on the same dataset.
# 
# This is designed to be run using the `run_experiments.py` script so we get some config from envvars.

# In[ ]:

# Load parameters from env variables
import os
from datetime import datetime

EXP_NAME = os.environ.get('EXP_NAME', 'labels_4_test_fold_0_rep_0')
DEVICE = os.environ.get('DEVICE', 'gpu0')

OUTDIR = os.environ.get(
    'OUTDIR',
    '../_out/%s_%s_%s_rot90' % (EXP_NAME, DEVICE, datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
)

CNN_NEPOCHS = int(os.environ.get('CNN_NEPOCHS', 50))
HISTNN_NEPOCHS = int(os.environ.get('HISTNN_NEPOCHS', 50))
MERGED_NEPOCHS = int(os.environ.get('MERGED_NEPOCHS', 50))
KERAS_VERBOSE = int(os.environ.get('KERAS_VERBOSE', 1))

print 'EXP_NAME :', EXP_NAME
print 'DEVICE :', DEVICE
print 'CNN_NEPOCHS : ', CNN_NEPOCHS
print 'HISTNN_NEPOCHS : ', HISTNN_NEPOCHS
print 'MERGED_NEPOCHS : ', MERGED_NEPOCHS
print 'OUTDIR : ', OUTDIR


# In[ ]:

# Theano config
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=%s,nvcc.fastmath=True,floatX=float32' % DEVICE


# In[ ]:

import time
import numpy as np
import agronn.dataio as dataio
import agronn.utils as utils
import agronn.keras_utils as keras_utils
import pylab as pl
#get_ipython().magic(u'matplotlib inline')


# In[ ]:

start_time = time.time()


# In[ ]:

utils.mkdir_p(OUTDIR)


# In[ ]:

# This is a dict where we'll store our results for later analysis
log = {}


# In[ ]:

winsize = 21
nbins = 20
# we work on centered RGB images
binranges = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]


# In[ ]:

mosaic, id2label, train_ij, test_ij, y_train, y_test, d = dataio.load(EXP_NAME, ret_d=True,
                                                                      data_fname='data_rot90_final.hdf5')
img = d['img']


# In[ ]:

print train_ij.shape
print test_ij.shape


# In[ ]:

print "mosaic shape ", mosaic.shape
#pl.imshow(img_mosaic[10000:15000])

print "mosaic takes ", mosaic.nbytes / (1024. * 1024.), "mb"


# In[ ]:

scaler = utils.ImageCenterer().fit(img)
mosaic_scaled = scaler.transform(mosaic).astype(np.float32)


# In[ ]:

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.layer_utils import print_layer_shapes
from keras.utils import np_utils
import keras

import keras.models as models

#weights_init = 'uniform'
weights_init = 'glorot_uniform'
activation = 'relu'
nclasses = len(id2label)

batch_size = 256

Y_train = np_utils.to_categorical(y_train, nclasses)


# In[ ]:

from sklearn.metrics import classification_report

def eval_classif(m, ij, y):
    y_pred = m.predict({'ij':ij})['output'].argmax(axis=1)
    report = 'kappa : %f\n' % utils.kappa(y, y_pred)
    report += classification_report(y, y_pred, labels=np.arange(nclasses),
                                    target_names=id2label)
    return report, y_pred

def eval_model(m):
    # Don't evaluate on whole train, this takes too long
    samples = np.random.choice(train_ij.shape[0], 20000)
    train_report, y_train_pred = eval_classif(m, train_ij[samples], y_train[samples])    
    test_report, y_test_pred = eval_classif(m, test_ij, y_test)
    
    d = {
        'train_report' : train_report,
        'test_report' : test_report,
        'y_train_pred' : y_train_pred,
        'y_train_true' : y_train[samples],
        'y_test_pred' : y_test_pred,
        'y_test_true' : y_test
    }
    return d

def print_model_eval(m):
    """
    Returns y_pred_train, y_true_train, y_pred_test, y_true_test
    """
    d = eval_model(m)
    print '-- train (sampled)'
    print d['train_report']

    print '-- test'
    print d['test_report']
    return d
    


# In[ ]:

# Share the ExtractWindowsLayer because it takes GPU memory
extract_win_layer = keras_utils.ExtractWindowsLayer(winsize, mosaic_scaled)


# This is how you'd do to extract the windows once, but this doesn't seem to change the training time much :
# 
# 
#     import agronn.my_ops as my_ops
#     import theano.tensor
# 
#     img_t = theano.tensor.tensor3()
#     win_ij_t = theano.tensor.matrix(dtype='int32')
# 
#     windows_2d = my_ops.extract_2d_windows(winsize)(img_t, win_ij_t)
#     extract_windows = theano.function(
#         [img_t, win_ij_t],
#         windows_2d,
#     )
# 
#     train_win = extract_windows(mosaic_scaled, train_ij)
#     train_win = np.array(out_win)
# 
#     print train_win.shape
#     print train_win.nbytes / (1024. * 1024.), ' mb'

# # CNN

# In[ ]:

cnn_pretrain = models.Graph()
cnn_pretrain.scaler = scaler
cnn_pretrain.add_input(name='ij', input_shape=(2,))

cnn_pretrain.add_node(extract_win_layer, name='extract_windows', input='ij')
#cnn_pretrain.add_input(name='extract_windows', input_shape=(3, winsize, winsize))

# cnn
cnn = models.Graph()
cnn.add_input(name='win', input_shape=(3, winsize, winsize))
cnn.add_node(Convolution2D(48, 11, 11, border_mode='valid', init=weights_init),
             name='conv1', input='win')
cnn.add_node(Activation(activation), name='act1', input='conv1')
cnn.add_node(MaxPooling2D(pool_size=(2,2)), name='pool1', input='act1')
cnn.add_node(Dropout(0.25), name='dropout1', input='pool1')

cnn.add_node(Convolution2D(48, 3, 3, border_mode='valid', init=weights_init), 
             name='conv2', input='dropout1')
cnn.add_node(Activation(activation), name='act2', input='conv2')
cnn.add_node(MaxPooling2D(pool_size=(2,2)), name='pool2', input='act2')
cnn.add_node(Dropout(0.25), name='dropout2', input='pool2')
cnn.add_node(Flatten(), name='flatten', input='dropout2')
cnn.add_node(Dense(128, init=weights_init), name='dense1',
                      input='flatten')
cnn.add_node(Activation(activation), name='act3', input='dense1')
cnn.add_node(Dropout(0.1), name='dropout3', input='act3')
cnn.add_output(name='cnn_output', input='dropout3')

print_layer_shapes(cnn, input_shapes={'win':(42, 3, 21, 21)})

# -- pred
cnn_pretrain.add_node(cnn, name='cnn', input='extract_windows')
cnn_pretrain.add_node(Dense(nclasses, init=weights_init), name='dense2',
                      input='cnn')
cnn_pretrain.add_node(Activation('softmax'), name='softmax', input='dense2')
cnn_pretrain.add_output(name='output', input='softmax')

cnn_pretrain.compile('adam', {'output':'categorical_crossentropy'})

print_layer_shapes(cnn_pretrain, input_shapes={'ij':train_ij[:42].shape})
#print_layer_shapes(cnn_pretrain, input_shapes={'extract_windows':train_win[:42].shape})

keras_utils.print_model_params(cnn_pretrain)

cnn_pretrain_history = cnn_pretrain.fit(
    {'ij': train_ij, 'output': Y_train},
    #{'extract_windows' : train_win, 'output': Y_train},
    batch_size=batch_size, nb_epoch=CNN_NEPOCHS,
    verbose=KERAS_VERBOSE, shuffle=True
)

pl.figure()
pl.title('Training loss')
pl.plot(cnn_pretrain_history.epoch,
        cnn_pretrain_history.history['loss'], label='loss')
pl.savefig(os.path.join(OUTDIR, 'cnn_train.png'))

print 'done'


# In[ ]:

cnn_eval = print_model_eval(cnn_pretrain)
log['cnn'] = {
    'eval' : cnn_eval,
    'history' : cnn_pretrain_history.history
}
with open(os.path.join(OUTDIR, 'cnn_train_score.txt'), 'w') as f:
    f.write(cnn_eval['train_report'])
with open(os.path.join(OUTDIR, 'cnn_test_score.txt'), 'w') as f:
    f.write(cnn_eval['test_report'])


# In[ ]:

keras_utils.save_model(os.path.join(OUTDIR, 'cnn.zip'), cnn_pretrain)


# In[ ]:

# Save cnn weights
weights = cnn_pretrain.get_weights()
print [w.shape for w in weights]

wimg = utils.make_mosaic_rgb(weights[0].transpose(0, 2, 3, 1), 7, 7)
wimg[~wimg.mask] = utils.norm01(wimg[~wimg.mask])
wimg[wimg.mask] = 0
pl.figure(figsize=(10,10))
pl.imshow(wimg, interpolation='nearest')
pl.savefig(os.path.join(OUTDIR, 'cnn_weights.png'), dpi=150)


# # histnn

# In[ ]:

histnn_pretrain = models.Graph()
histnn_pretrain.scaler = scaler
histnn_pretrain.add_input(name='ij', input_shape=(2,))

histnn_pretrain.add_node(extract_win_layer, name='extract_windows', input='ij')

# histnn
histnn = models.Graph()
histnn.add_input(name='win', input_shape=(3, winsize, winsize))
histnn.add_node(keras_utils.ExtractHist1DLayer(nbins, binranges),
           name='extract_hist', input='win')
histnn.add_node(Flatten(), name='flatten', input='extract_hist')
histnn.add_node(Dense(32, init=weights_init), name='dense1', input='flatten')
histnn.add_node(Activation(activation), name='act1', input='dense1')
histnn.add_node(Dropout(0.1), name='dropout1', input='act1')

histnn.add_node(Dense(32, init=weights_init), name='dense2', input='dropout1')
histnn.add_node(Activation(activation), name='act2', input='dense2')
histnn.add_node(Dropout(0.25), name='dropout2', input='act2')
histnn.add_output(name='histnn_output', input='dropout2')

# -- pred
histnn_pretrain.add_node(histnn, name='histnn', input='extract_windows')
histnn_pretrain.add_node(Dense(nclasses, init=weights_init), name='dense2',
                         input='histnn')
histnn_pretrain.add_node(Activation('softmax'), name='softmax', input='dense2')
histnn_pretrain.add_output(name='output', input='softmax')

histnn_pretrain.compile('adam', {'output':'categorical_crossentropy'})

print_layer_shapes(histnn_pretrain, input_shapes={'ij':train_ij[:42].shape})

keras_utils.print_model_params(histnn_pretrain)

histnn_pretrain_history = histnn_pretrain.fit(
    {'ij': train_ij, 'output': Y_train},
    batch_size=batch_size, nb_epoch=HISTNN_NEPOCHS,
    verbose=KERAS_VERBOSE, shuffle=True
)

pl.figure()
pl.title('Training loss')
pl.plot(histnn_pretrain_history.epoch,
        histnn_pretrain_history.history['loss'], label='loss')
pl.savefig(os.path.join(OUTDIR, 'histnn_train.png'))

print 'done'


# In[ ]:

histnn_eval = print_model_eval(histnn_pretrain)
log['histnn'] = {
    'eval' : histnn_eval,
    'history' : histnn_pretrain_history.history
}
with open(os.path.join(OUTDIR, 'histnn_train_score.txt'), 'w') as f:
    f.write(histnn_eval['train_report'])
with open(os.path.join(OUTDIR, 'histnn_test_score.txt'), 'w') as f:
    f.write(histnn_eval['test_report'])


# In[ ]:

keras_utils.save_model(os.path.join(OUTDIR, 'histnn.zip'), histnn_pretrain)


# # Merged

# In[ ]:

merged_nn = models.Graph()
merged_nn.scaler = scaler
merged_nn.add_input(name='ij', input_shape=(2,))

merged_nn.add_node(extract_win_layer, name='extract_windows', input='ij')

merged_nn.add_node(histnn, name='histnn', input='extract_windows')
merged_nn.add_node(cnn, name='cnn', input='extract_windows')

merged_nn.add_node(Dense(128, init=weights_init), name='dense_merge_1',
                   inputs=['histnn', 'cnn'], merge_mode='concat')
merged_nn.add_node(Activation(activation), name='dense_act_1', input='dense_merge_1')
merged_nn.add_node(Dropout(0.25), name='merge_dropout1', input='dense_act_1')
merged_nn.add_node(Dense(nclasses, init=weights_init), name='dense2',
                   input='merge_dropout1')
merged_nn.add_node(Activation('softmax'), name='softmax', input='dense2')
merged_nn.add_output(name='output', input='softmax')

# Finetune by retraining the whole network with a low learning rate
#opt = keras.optimizers.Adam(lr=0.0001)
opt = keras.optimizers.Adam(lr=0.001)
merged_nn.compile(opt, {'output':'categorical_crossentropy'})

print_layer_shapes(merged_nn, input_shapes={'ij':train_ij[:42].shape})

keras_utils.print_model_params(merged_nn)

merged_nn_history = merged_nn.fit(
    {'ij': train_ij, 'output': Y_train},
    batch_size=batch_size, nb_epoch=MERGED_NEPOCHS,
    verbose=KERAS_VERBOSE, shuffle=True
)

pl.figure()
pl.title('Training loss')
pl.plot(merged_nn_history.epoch,
        merged_nn_history.history['loss'], label='loss')
pl.savefig(os.path.join(OUTDIR, 'merged_train.png'))

print 'done'


# In[ ]:

merged_eval = print_model_eval(merged_nn)
log['merged'] = {
    'eval' : merged_eval,
    'history' : merged_nn_history.history
}
with open(os.path.join(OUTDIR, 'merged_train_score.txt'), 'w') as f:
    f.write(merged_eval['train_report'])
with open(os.path.join(OUTDIR, 'merged_test_score.txt'), 'w') as f:
    f.write(merged_eval['test_report'])


# In[ ]:

keras_utils.save_model(os.path.join(OUTDIR, 'merged_nn.zip'), merged_nn)


# In[ ]:

# Save merged nn weights
weights = merged_nn.get_weights()
print [w.shape for w in weights]

wimg = utils.make_mosaic_rgb(weights[4].transpose(0, 2, 3, 1), 7, 7)
wimg[~wimg.mask] = utils.norm01(wimg[~wimg.mask])
wimg[wimg.mask] = 0
pl.figure(figsize=(10,10))
pl.imshow(wimg, interpolation='nearest')
pl.savefig(os.path.join(OUTDIR, 'merged_weights.png'), dpi=150)


# In[ ]:

elapsed = time.time() - start_time
print 'took %f [s]' % elapsed
log['elapsed_time_s'] = elapsed


# In[ ]:

utils.pickle_save(os.path.join(OUTDIR, 'log.pickle'), log)

