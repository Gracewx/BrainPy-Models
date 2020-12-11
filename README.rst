BrainPy-Models
===================

**Note**: *We welcome your contributions for model implementations.*


``BrainPy-Models`` is a repository accompany with 
`BrainPy <https://github.com/PKU-NIP-Lab/BrainPy>`_, 
which is a framework for spiking neural network simulation. 
With BrainPy, we implements the most canonical and
effective neuron models and synapse models,
and show them in ``BrainPy-Models``.

Here, users can directly import our models into your network,
and also can learn examples of how to use BrainPy from 
`Documentations <https://brainpy-models.readthedocs.io/en/latest/>`_.



We provide the following models:

+---------------------------------+---------------------------------+-------------------+----------------------------+
|   Neuron models                 |   Synapse models                |   Learning rules  | Networks                   |
+=================================+=================================+===================+============================+
| Leaky integrate-and-fire model  | Alpha Synapse                   |   STDP            |Continuous attractor network|
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Hodgkin-Huxley model            | AMPA / NMDA                     |                   |    E/I balance network     |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Izhikevich model                | GABA_A / GABA_B                 |                   |                            | 
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Morris–Lecar model              | Exponential Decay Synapse       |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Generalized integrate-and-fire  | Difference of Two Exponentials  |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Exponential integrate-and-fire  |                                 |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Hindmarsh–Rose model            |                                 |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Wilson-Cowan model              |                                 |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
|                                 |                                 |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+





Installation
============

Install from source code::

    > python setup.py install


Install ``BrainPy-Models`` using ``conda``::

    > conda install -c brainpy bpmodels


Install ``BrainPy-Models`` using ``pip``::

    > pip install bpmodels


The following packages need to be installed to use ``BrainPy-Models``:

- Python >= 3.7
- Matplotlib >= 2.0
- BrainPy


Quick Start
============

The use of ``bpmodels`` is very convenient, let's take an example of exploring different firing types in the izhikevich model.

We start by importing the ``brainpy`` and ``bpmodels`` packages and set profile.

::

    import brainpy as bp
    import bpmodels

    # set profile
    bp.profile.set(backend='numba',
                device='cpu',
                merge_steps=True,
                numerical_method='exponential')


The pre-defined izhikevich model provides many different firing types, so we can use ``type='tonic spiking'`` to get a pre-defined model with tonic spiking parameters.

::

    izh = bpmodels.neurons.get_Izhikevich(type='tonic spiking')
    
    (step_I, duration) = bp.inputs.constant_current(
                            [(0, 50), (10, 100), (0, 50)])

    neuron = bp.NeuGroup(neu_type, 1, monitors= ['V', 'u'])

    neuron.run(duration = duration, inputs = ["ST.input", step_I])
    

    # visualization

    ts = neuron.mon.ts
    fig, gs = bp.visualize.get_figure(3, 1, 3, 6)

    fig.add_subplot(gs[0, 0])
    plt.plot(ts, step_I, 'r')
    plt.ylabel('Input Current')
    plt.xlim(-0.1, duration + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('Input Current')

    fig.add_subplot(gs[1, 0])
    plt.plot(ts, neuron.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlabel('Time (ms)')
    plt.xlim(-0.1, duration + 0.1)
    plt.ylim(-95., 40.)
    plt.title('Membrane potential')
    plt.legend()

    fig.add_subplot(gs[2, 0])
    plt.plot(ts, neuron.mon.u[:, 0], label='u')
    plt.xlim(-0.1, duration + 0.1)
    plt.ylabel('Recovery variable')
    plt.xlabel('Time (ms)')
    plt.title('Recovery variable')
    plt.legend()

    plt.show()
    
    
Then you would expect to see the following output:

.. image:: docs/images/izh_tonic_spiking.png
