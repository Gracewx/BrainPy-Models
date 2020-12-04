.. BrainPy-Models documentation master file, created by
   sphinx-quickstart on Sat Oct 17 15:33:12 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BrainPy-Models documentation
===============================

``BrainPy-Models`` is based on `BrainPy <https://brainpy.readthedocs.io/>`_ neuronal dynamics simulation framework. Here you can find neurons, synapses models and topological networks implemented with BrainPy.

The prior goal of ``BrainPy-Models`` is to free users from repeatly implementing the most simple and commonly used models, instead they can import ``get_*()`` functions and take the advantage of our models. 

``BrainPy-Models`` is also designed to display ``BrainPy``. We hope users can learn in our models and examples how to use ``BrainPy`` (better), how to understand it, and the most elegant and appealing features of it.

Enjoy your diving in the sea of dynamical models!


.. note::

   We welcome your implementation about `neurons`, `synapses`, `learning rules`,
   `networks` and `paper examples`. https://github.com/PKU-NIP-Lab/BrainPy-Models

The documentation includes three main parts.

In `Tutorials` section, you can learn mechanisms and implementations of several classical neuron models, synapse models and learning rules. With ``BrainPy``, we want to reveal the world of computational neuroscience to new learners here.

In `APIs` section, we give details about all the provided models, present their mathematical representations, explan their APIs and give biologically plausible meaning to parameters and members.

In `Examples` section, we provide examples using our APIs and implentations of widely known and used 
networks (ex. Excitatory-inhibitory balanced network). Some of them are implemented totally based on our API, others show how easy and flexible can users define models using ``BrainPy``. We hope users can see these as examples of using `bpmodels` to simplify network implementation.



.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   
   tutorials/neurons
   tutorials/synapses

.. toctree::
   :maxdepth: 2
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
