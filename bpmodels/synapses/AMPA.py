# -*- coding: utf-8 -*-

import brainpy as bp
from brainpy import numpy as np


def get_AMPA1_scalar(g_max=0.10, E=0., tau_decay=2.0):
    """AMPA conductance-based synapse (type 1).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})

    Parameters
    ----------
    g_max : float
        Maximum conductance.
    E : float
        Reversal potential.
    tau_decay : float
        Tau for decay.
    """

    requires = dict(
        ST=bp.types.SynState(['s'], help='AMPA synapse state.'),
        pre=bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "spike" item.'),
        post=bp.types.NeuState(['V', 'input'], help='Post-synaptic neuron state must have "V" and "input" item.'),
    )

    @bp.integrate
    def ints(s, _t_):
        return - s / tau_decay

    def update(ST, _t_, pre):
        s = ints(ST['s'], _t_)
        s += pre['spike']
        ST['s'] = s

    @bp.delayed
    def output(ST, post):
        I_syn = - g_max * ST['s'] * (post['V'] - E)
        post['input'] += I_syn

    return bp.SynType(name='AMPA_synapse',
                      requires=requires,
                      steps=(update, output),
                      vector_based=False)


def get_AMPA1(g_max=0.10, E=0., tau_decay=2.0):
    """AMPA conductance-based synapse (type 1).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})

    Parameters
    ----------
    g_max : float
        Maximum conductance.
    E : float
        Reversal potential.
    tau_decay : float
        Tau for decay.
    """

    @bp.integrate
    def ints(s, _t_):
        return - s / tau_decay

    # requirements
    # ------------

    requires = {
        'ST': bp.types.SynState(['s', 'g'], help='AMPA synapse state.'),
        'pre': bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "spike" item.'),
        'post': bp.types.NeuState(['V', 'input'], help='Post-synaptic neuron state must have "V" and "input" item.'),
        'pre2syn': bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index'),
        'post2syn': bp.types.ListConn(help='Post-synaptic neuron index -> synapse index'),
    }

    # model logic
    # -----------

    def update(ST, _t_, pre, pre2syn):
        s = ints(ST['s'], _t_)
        spike_idx = np.where(pre['spike'] > 0.)[0]
        for i in spike_idx:
            syn_idx = pre2syn[i]
            s[syn_idx] += 1.
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post, post2syn):
        g = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            g[post_id] = np.sum(ST['g'][syn_ids])
        post['input'] -= g * (post['V'] - E)

    return bp.SynType(name='AMPA_synapse',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)


def get_AMPA2_scalar(g_max=0.42, E=0., alpha=0.98, beta=0.18, T=0.5, T_duration=0.5):
    """AMPA conductance-based synapse (type 2).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{ds}{dt} &=\\alpha[T](1-s)-\\beta s

    Parameters
    ----------
    g_max : float
        Maximum conductance.
    E : float
        Reversal potential.
    alpha : float
        Binding constant.
    beta : float
        Unbinding constant.
    T : float
        Neurotransmitter.
    T_duration : float
        Binding of neurotransmitter.
    """

    requires = {
        'ST': bp.types.SynState({'s': 0., 't_last_spike': -1e7},
                                help=""" "s": Synaptic state.
                                    "t_last_spike": Pre-synaptic neuron spike time.
                                """),
        'pre': bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "spike" item.'),
        'post': bp.types.NeuState(['V', 'input'], help='Post-synaptic neuron state must have "V" and "input" item.'),
    }

    @bp.integrate
    def int_s(s, _t_, TT):
        return alpha * TT * (1 - s) - beta * s

    def update(ST, _t_, pre):
        if pre['spike'] > 0.:
            ST['t_last_spike'] = _t_
        TT = ((_t_ - ST['t_last_spike']) < T_duration) * T
        s = np.clip(int_s(ST['s'], _t_, TT), 0., 1.)
        ST['s'] = s

    @bp.delayed
    def output(ST, post):
        post_val = - g_max * ST['s'] * (post['V'] - E)
        post['input'] += post_val

    return bp.SynType(name='AMPA_synapse',
                      requires=requires,
                      steps=(update, output),
                      vector_based=False)


def get_AMPA2(g_max=0.42, E=0., alpha=0.98, beta=0.18, T=0.5, T_duration=0.5):
    """AMPA conductance-based synapse (type 2).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{ds}{dt} &=\\alpha[T](1-s)-\\beta s

    Parameters
    ----------
    g_max : float
        Maximum conductance.
    E : float
        Reversal potential.
    alpha : float
        Binding constant.
    beta : float
        Unbinding constant.
    T : float
        Neurotransmitter.
    T_duration : float
        Binding of neurotransmitter.
    """

    @bp.integrate
    def int_s(s, _t_, TT):
        return alpha * TT * (1 - s) - beta * s

    requires = dict(
        ST=bp.types.SynState({'s': 0., 't_last_spike': -1e7, 'g': 0.},
                             help='AMPA synapse state.\n'
                                  '"s": Synaptic state.\n'
                                  '"t_last_spike": Pre-synaptic neuron spike time.'),
        pre=bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "spike" item.'),
        post=bp.types.NeuState(['V', 'input'], help='Post-synaptic neuron state must have "V" and "input" item.'),
        pre2syn=bp.types.ListConn(
            help='Pre-synaptic neuron index -> synapse index'),
        post2syn=bp.types.ListConn(
            help='Post-synaptic neuron index -> synapse index'),
    )

    def update(ST, _t_, pre, pre2syn):
        for i in np.where(pre['spike'] > 0.)[0]:
            syn_idx = pre2syn[i]
            ST['t_last_spike'][syn_idx] = _t_
        TT = ((_t_ - ST['t_last_spike']) < T_duration) * T
        s = np.clip(int_s(ST['s'], _t_, TT), 0., 1.)
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['input'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='AMPA_synapse',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)
