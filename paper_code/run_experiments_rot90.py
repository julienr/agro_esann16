#!/usr/bin/env python
import os
import time
import sys
from os import environ
import subprocess
from subprocess import call, check_output, check_call, CalledProcessError
from datetime import datetime


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST:
            pass
        else: raise

experiments = []
for i in xrange(10):
    #experiments.append('labels_4_test_fold_0_rep_%d' % i)
    experiments.append('labels_6_test_fold_0_rep_%d' % i)

device = 'gpu0'
cnn_nepochs = 60
histnn_nepochs = 60
merged_nepochs = 60

for i, exp in enumerate(experiments):
    print '\n\n---- Experiment ', i, ' on ', len(experiments)
    starttime = time.time()
    timestr = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    outdir = '../_out/rot90_final/%s_%s_%s_epochs_%d_%d_%d' % (exp, device,
        timestr, cnn_nepochs, histnn_nepochs, merged_nepochs)
    mkdir_p(outdir)
    os.environ['EXP_NAME'] = exp
    os.environ['DEVICE'] = device
    os.environ['KERAS_VERBOSE'] = "2"
    os.environ['OUTDIR'] = outdir

    os.environ['CNN_NEPOCHS'] = str(cnn_nepochs)
    os.environ['HISTNN_NEPOCHS'] = str(histnn_nepochs)
    os.environ['MERGED_NEPOCHS'] = str(merged_nepochs)

    logfile = os.path.join(outdir, 'log.txt')
    print 'Starting %s' % outdir
    print 'Logging to %s' % logfile
    args = [sys.executable, 'train_rot90.py']
    try:
        output = check_output(args)
    except CalledProcessError as e:
        # For some strange reason, train.py always finishes with
        # retcode -11 even though the training suceed. So just ignore
        print 'Finished with error (retcode=%d)' % e.returncode
        output = e.output
        #print 'output :\n', e.output
        #sys.exit(-1)

    with open(logfile, 'w') as f:
        f.write(output)

    elapsed = time.time() - starttime
    print 'took %f [s]' % elapsed
