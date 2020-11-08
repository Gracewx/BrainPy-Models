.. BrainPy-Models documentation master file, created by
   sphinx-quickstart on Sat Oct 17 15:33:12 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BrainPy-Models documentation
===============================

``BrainPy-Models`` is based on `BrainPy <https://brainpy.readthedocs.io/>`_ neuronal
dynamics simulation framework. Here you can find the list of all the models implemented with
BrainPy for neurons, synapses and topological networks.



.. note::

   We welcome your implementation about `neurons`, `synapses`, `learning rules`,
   `networks` and `paper examples`. https://github.com/PKU-NIP-Lab/BrainPy-Models

The documentation includes three parts.

In `Tutorials` section, you can learn classical neuron models, synapse models 
and implementations of learning rules. With ``BrainPy``, 
our goal is to reveal the world of computational neuroscience to new learners.

In `Examples` section, you can see some examples of using `bpmodels` to simplify 
networks implementation.

And in `APIs` section, you can see details about all the models that we provided 
and how to use them.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   
   tutorials/neurons
   tutorials/synapses
   tutorials/learning_rules

.. toctree::
   :maxdepth: 1
   :caption: Examples
   
   examples/EI_balanced

.. toctree::
   :maxdepth: 1
   :caption: APIs
   
   apis/neurons
   apis/synapses

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
