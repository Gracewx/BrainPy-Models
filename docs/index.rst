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

In `Tutorials` section, you can learn implementations of classical neuron 
models, synapse models and learning rules. With ``BrainPy``, our goal is to 
reveal the world of computational neuroscience to new learners.

In `Examples` section, we provide implentations of widely known and used 
networks (ex. Excitatort-inhibitory balanced network). We hope users can see 
these as examples of using `bpmodels` to simplify network implementation.

In `APIs` section, we give details about all the models that we provided and how
to use them with our API.



.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   
   tutorials/neurons
   tutorials/synapses
   tutorials/learning_rules

.. toctree::
   :maxdepth: 1
   :caption: Examples
   
   examples/neurons
   examples/synapses
   examples/learning_rules
   examples/networks

.. toctree::
   :maxdepth: 2
   :caption: API documentation
   
   apis/neurons
   apis/synapses
   apis/learning_rules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
