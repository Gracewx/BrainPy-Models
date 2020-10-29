BrainPy-Models
===================

**Note**: *BrainPy-Models is a project under development.*
*More features are coming soon. Contributions are welcome.*


``BrainPy-Models`` is a repository accompany with `BrainPy <https://github.com/PKU-NIP-Lab/BrainPy>`_, which is a framework for spiking neural network simulation. With BrainPy, we implements the most cononical and effective neuron models and synapse models, and show them in ``BrainPy-Models``.

Here, users can learn examples of how to use BrainPy from `Documentations <https://brainpy-models.readthedocs.io/en/latest/>`_, and directly import our models into your network.


Installation
============


Install from source code:
::

    > git clone https://github.com/PKU-NIP-Lab/BrainPy-Models
    > python setup.py install
    > # or
    > pip install git+https://github.com/PKU-NIP-Lab/BrainPy-Models

The following packages need to be installed to use ``BrainPy-Models``:

- Python >= 3.5

- NumPy >= 1.13

- Sympy >= 1.2

- Matplotlib >= 2.0

- autopep8

- BrainPy


Use a Hodgkin–Huxley neuron model
====================================

Let's start with importing ``brainpy`` and ``bpmodels``, and set global profile.

::

  import brainpy as bp
  from bpmodels.neurons import HH

  # use "profile" to make global settings.
  bp.profile.set(backend='numpy', dt=0.02, numerical_method='milstein', merge_steps=True)

To quickly run simulation of a HH neuron model, you can call ``HH.simulate()``.

::

  (ts, V, m, h, n) = HH.simulate(input_current = 5., duration = 80., geometry=(1,))

If you want to be more flexible, you can use ``HH.get_neuron()`` to get an ``NeuType`` object to run simulation.

::

  HH_neuron = HH.get_neuron(geometry=1, noise=0.)
  HH_neuron.run(duration=duration, inputs=['ST.input', 5.], report=True)

  ts = HH_neuron.mon.ts
  V = HH_neuron.mon.V[:, 0]
  m = HH_neuron.mon.m[:, 0]
  h = HH_neuron.mon.h[:, 0]
  n = HH_neuron.mon.n[:, 0]
  
It is also possible to directly call ``HH.define_HH()`` if necessary.

::

  HH = HH.define_HH(noise = 0.)
  HH_neuron = bp.NeuGroup(HH, geometry=1, 
                      monitors=['spike', 'V', 'm', 'h', 'n'])

  HH_neuron.run(duration=duration, inputs=['ST.input', 5.], report=True)

  ts = HH_neuron.mon.ts
  V = HH_neuron.mon.V[:, 0]
  m = HH_neuron.mon.m[:, 0]
  h = HH_neuron.mon.h[:, 0]
  n = HH_neuron.mon.n[:, 0]
