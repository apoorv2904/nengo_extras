************
Nengo extras
************

Extra utilities and add-ons for Nengo.

This repository contains utilities that occupy
a liminal space not quite generic enough for inclusion in Nengo_,
but useful enough that they should be publicly accessible.

Some of these utilities may eventually migrate to Nengo_,
and others may be split off into their own separate repositories.

.. _Nengo: https://github.com/nengo/nengo


Configuration
=============

This repository makes use of Nengo's RC file system to store configuration
information. See the Nengo_ documentation for how to add to the configuration
file. Configuration settings are listed below.

.. code:: bash

    [nengo_extras]
    # directory to store downloaded datasets
    data_dir = ~/data


Summary of Contents
===================

The current contents of this repository are listed below.

Additional networks
-------------------

- ``nengo_extras.networks.MatrixMult``

Convolutional Network processes
-------------------------------

- ``nengo_extras.Conv2d``
- ``nengo_extras.Pool2d``

Neuron types
------------

- ``nengo_extras.FastLIF``
- ``nengo_extras.SoftLIFRate``
