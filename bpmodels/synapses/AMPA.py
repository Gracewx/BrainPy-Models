# -*- coding: utf-8 -*-

import brainpy as bp
from brainpy import numpy as np

def get_AMPA1(g_max=0.10, E=0., tau_decay=2.0):
    """AMPA conductance-based synapse (type 1).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})


    ST refers to the synapse state, items in ST are listed below:
    
    =============== ================== =========================================================
    **Member name** **Initial values** **Explanation**
    --------------- ------------------ ---------------------------------------------------------
    s                   0               Gating variable.
    
    g                   0               Synapse conductance.
    =============== ================== =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        g_max (float): Maximum conductance in µmho (µS).
        E (float): The reversal potential for the synaptic current.
        tau_decay (float): The time constant of decay.

    Returns:
        bp.Neutype

    References:
        .. [1] Brunel N, Wang X J. Effects of neuromodulation in a cortical network 
                model of object working memory dominated by recurrent inhibition[J]. 
                Journal of computational neuroscience, 2001, 11(1): 63-85.
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
                      mode = 'vector')



def get_AMPA2(g_max=0.42, E=0., alpha=0.98, beta=0.18, T=0.5, T_duration=0.5):
    """AMPA conductance-based synapse (type 2).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{ds}{dt} &=\\alpha[T](1-s)-\\beta s

    ST refers to the synapse state, items in ST are listed below:
    
    ================ ================== =========================================================
    **Member name**  **Initial values** **Explanation**
    ---------------- ------------------ ---------------------------------------------------------
    s                 0                 Gating variable.
    
    g                 0                 Synapse conductance.

    t_last_pre_spike  -1e7              Last spike time stamp of the pre-synaptic neuron.
    ================ ================== =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        g_max (float): Maximum conductance in µmho (µS).
        E (float): The reversal potential for the synaptic current.
        alpha (float): Binding constant.
        beta (float): Unbinding constant.
        T (float): Neurotransmitter binding coefficient.
        T_duration (float): Duration of the binding of neurotransmitter.

    Returns:
        bp.Neutype

    References:
        .. [1] Vijayan S, Kopell N J. Thalamic model of awake alpha oscillations 
                and implications for stimulus processing[J]. Proceedings of the 
                National Academy of Sciences, 2012, 109(45): 18553-18558.
    """

    @bp.integrate
    def int_s(s, _t_, TT):
        return alpha * TT * (1 - s) - beta * s

    requires = dict(
        ST=bp.types.SynState({'s': 0., 't_last_pre_spike': -1e7, 'g': 0.},
                             help='AMPA synapse state.\n'
                                  '"s": Synaptic state.\n'
                                  '"t_last_pre_spike": Pre-synaptic neuron spike time.'),
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
            ST['t_last_pre_spike'][syn_idx] = _t_
        TT = ((_t_ - ST['t_last_pre_spike']) < T_duration) * T
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
                      mode = 'vector')
