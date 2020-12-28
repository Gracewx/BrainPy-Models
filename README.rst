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
| Hodgkin-Huxley model            | AMPA / NMDA                     |   BCM rule        |    E/I balance network     |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Izhikevich model                | GABA_A / GABA_B                 |   Oja's rule      |   gamma oscillations       | 
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Morris–Lecar model              | Exponential Decay Synapse       |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Generalized integrate-and-fire  | Difference of Two Exponentials  |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Exponential integrate-and-fire  | Short-term plasticity           |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Quadratic integrate-and-fire    | Gap junction                    |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| adaptive Exponential IF         | Voltage jump                    |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| adaptive Quadratic IF           |                                 |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Hindmarsh–Rose model            |                                 |                   |                            |
+---------------------------------+---------------------------------+-------------------+----------------------------+
| Wilson-Cowan model              |                                 |                   |                            |
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

The use of ``bpmodels`` is very convenient, let's take an example of the implementation of the E-I balanced network.

We start by importing the ``brainpy`` and ``bpmodels`` packages and set profile.

::

    import brainpy as bp
    import bpmodels
    import brainpy.numpy as np
    import matplotlib.pyplot as plt

    # set profile
    bp.profile.set(jit=True,
                device='cpu',
                merge_steps=True,
                numerical_method='exponential')

    # set parameters
    V_rest = -52.
    V_reset = -60.
    V_th = -50.
    num_exc = 500
    num_inh = 500
    prob = 0.15
    
    JE = 1 / np.sqrt(prob * num_exc)
    JI = 1 / np.sqrt(prob * num_inh)


The E-I balanced network is based on leaky Integrate-and-Fire (LIF) neurons 
connecting with single exponential decay synapses. As showed in the table above, 
``bpmodels`` provides pre-defined LIF neuron model and exponential synapse model, 
so we can use ``bpmodels.neurons.get_LIF`` and ``bpmodels.synapses.get_exponential`` 
to get the pre-defined models.

::

    neu = bpmodels.neurons.get_LIF(V_reset = V_reset, V_rest = V_rest, V_th=V_th, noise=0.)
    
    syn = bpmodels.synapses.get_exponential(tau_decay = 2.)

    # build network
    group = bp.NeuGroup(neu, geometry=num_exc + num_inh, monitors=['spike'])

    group.ST['V'] = np.random.random(num_exc + num_inh) * (V_th - V_rest) + V_rest
    
    exc_conn = bp.SynConn(syn,
                          pre_group=group[:num_exc],
                          post_group=group,
                          conn=bp.connect.FixedProb(prob=prob))
    exc_conn.ST['w'] = JE

    inh_conn = bp.SynConn(syn,
                          pre_group=group[num_exc:],
                          post_group=group,
                          conn=bp.connect.FixedProb(prob=prob))
    exc_conn.ST['w'] = -JI

    net = bp.Network(group, exc_conn, inh_conn)
    net.run(duration=1000., inputs=(group, 'ST.input', 3.))


    # visualization
    fig, gs = bp.visualize.get_figure(4, 1, 2, 12)

    fig.add_subplot(gs[:3, 0])
    bp.visualize.plot_raster(group.mon, net.ts, xlim=(50, 950))

    fig.add_subplot(gs[3, 0])
    rates = bp.measure.firing_rate(group.mon.spike, 5.)
    plt.plot(net.ts, rates)
    plt.xlim(50, 950)
    plt.show()
    
    
Then you would expect to see the following output:

.. image:: docs/images/EI_balanced.png
