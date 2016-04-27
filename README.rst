The agrovision dataset
======================

This is the agrovision dataset presented in our ESANN'16 `paper <cnn_histnn_esann16_paper.pdf>`_.

.. image:: _imgs/rgb.png
  :height: 300px
  :align: center

Welcome to the agrovision dataset webpage. This dataset is contains a high
resolution RGB image (5cm ground resolution) of an experimental farm field
containing 22 different crops.

The dataset also includes a digital surface model of the area.

We provide examples as jupyter/ipython notebooks.

Have a look at the `notebooks/1_plot.ipynb <notebooks/1_plot.ipynb>`_ notebook for an example on how
to load the dataset.

.. note::

  Please note that this dataset is free to use for research purposes. Please cite
  our paper if you use the dataset in your research. Also consider sending us an
  email to let us know what cool stuff you did :-)


loading the dataset in python
-----------------------------
The ``data/data.joblib`` file contains the image and the labels and the image
in a format that's easy to load from python using ``joblib.load``.


QGIS
----
The ``data/qgis`` folder contains the tif for the RGB and the DSM images as
well as geojson for the label polygons.

The ``notebooks/0_preprocess.ipynb`` notebook contains the code to convert
the QGIS files into the ``data/data.npz``.


Code for our paper
------------------
The code we used for our paper is available in the `paper_code` directory.
This includes training scripts for our CNN-HistNN and some custom keras layers
to perform Histograms extraction the GPU.

Contact
-------
Julien Rebetez, Héctor F. Satizabal, Matteo Mota, Dorothea Noll, Lucie Büchi,
Marina Wendling, Bertrand Cannelle, Andres Perez-Uribe and Stéphane Burgos