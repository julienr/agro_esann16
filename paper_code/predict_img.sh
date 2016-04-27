#!/bin/bash
IMG=~/agrovision/data/3_datasets/classif2/rasters/2013_10_08_rgb.tif

#EXPDIR=../results/rot90_final/labels_4_test_fold_0_rep_0_gpu0_2015_11_16_10_16_06_epochs_60_60_60/
EXPDIR=../results/rot90_final/labels_6_test_fold_0_rep_5_gpu0_2015_11_17_20_21_08_epochs_60_60_60/


for model in merged_nn.zip histnn.zip cnn.zip
do
    python run_model_on_image.py $EXPDIR/$model $IMG $EXPDIR/_pred_${model}
done
