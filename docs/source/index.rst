.. zokyo documentation master file, created by
   sphinx-quickstart on Wed Jan 12 17:03:48 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to zokyo's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Zokyo
========

Zokyo is a no-nonsense low-code computer vision augmentation library specifically built for automotive deep learning development which is
easy to integrate with your MLOps pipelines  

Look how easy it is to use:

.. code-block:: python

   from zokyo.augmentation import Builder
   pipeline = Builder(config_json="config.json")
   pipeline.process_and_save()

Installation
------------

Install Zokyo by running: 

.. code-block:: console

   make install

License
-------

The project is licensed under the Apache License 2.0.
