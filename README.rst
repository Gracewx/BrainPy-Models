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


Installation
============

Install from source code::

    > python setup.py git+https://github.com/PKU-NIP-Lab/BrainPy-Models


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

Let's see how to use ``bpmodels`` to quickly implement a network by seeing the example of E-I balanced network.

First import necessary packages and set parameters.

::

    import brainpy as bp
    import brainpy.numpy as np
    from bpmodels.neurons import get_LIF
    from bpmodels.synapses import get_alpha
    import matplotlib.pyplot as plt

    # set profile
    bp.profile.set(backend='numba',
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

Then you can get neuron and synapse models, for more details, see `APIs <https://brainpy-models.readthedocs.io/en/latest/>`_.

::

    # get neuron model
    neu = get_LIF(V_rest=V_rest, V_reset = V_reset, V_th=V_th, noise=0.)

    # get synapse model
    syn = get_alpha(tau_decay = 2.)

    # build network
    group = bp.NeuGroup(neu,
                        geometry=num_exc + num_inh,
                        monitors=['spike'])
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
    inh_conn.ST['w'] = -JI

    net = bp.Network(group, exc_conn, inh_conn)
    net.run(duration=1000., inputs=[(group, 'ST.input', 3.)], report=True)

    # visualization

    fig, gs = bp.visualize.get_figure(4, 1, 2, 12)

    fig.add_subplot(gs[:3, 0])
    bp.visualize.plot_raster(group.mon, net.ts, xlim=(50, 950))

    fig.add_subplot(gs[3, 0])
    rates = bp.measure.firing_rate(group.mon.spike, 5.)
    plt.plot(net.ts, rates)
    plt.xlim(50, 950)
    plt.show()
