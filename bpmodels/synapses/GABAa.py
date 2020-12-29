# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np
import sys

def get_GABAa1(g_max=0.4, E=-80., tau_decay=6., mode='vector'):
    """
    GABAa conductance-based synapse model (differential form).

    .. math::
        
        I_{syn}&= - \\bar{g}_{max} s (V - E)

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k}\\delta(t-t-{j}^{k})

    ST refers to synapse state, members of ST are listed below:
    
    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    s               0.                Gating variable.
    
    g               0.                Synapse conductance on post-synaptic neuron.
    =============== ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        g_max (float): Maximum synapse conductance.
        E (float): Reversal potential of synapse.
        tau_decay (float): Time constant of gating variable decay.
        mode (str): Data structure of ST members.

    Returns:
        bp.SynType: return description of GABAa synapse model.
    
    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """
    
    ST_vector = bp.types.SynState({'s': 0., 'g': 0.}, help = "GABAa synapse state")
    ST_scalar = bp.types.SynState(['s'], help = 'GABAa synapse state.')
    
    requires_vector = dict(
        pre=bp.types.NeuState(['spike'], help = "Pre-synaptic neuron state must have 'spike' item"),
        post=bp.types.NeuState(['V', 'input'], help = "Post-synaptic neuron state must have 'V' and 'input' item"),
        pre2syn=bp.types.ListConn(help="Pre-synaptic neuron index -> synapse index"),
        post2syn=bp.types.ListConn(help="Post-synaptic neuron index -> synapse index")
    )

    requires_scalar = {
        'pre': bp.types.NeuState(['spike'], help = 
                                 'Pre-synaptic neuron state must have "isFire"'),
        'post': bp.types.NeuState(['V', 'input'], help = 
                                 'Post-synaptic neuron state must include "input" and "Vr"')
    }

    @bp.integrate
    def int_s(s, t):
        return - s / tau_decay
    
    if mode=='scalar':
        def update(ST, _t, pre):
            s = int_s(ST['s'], _t)
            s += pre['spike']
            ST['s'] = s
    elif mode=='vector':
        def update(ST, pre, pre2syn):
            s = int_s(ST['s'], 0.)
            for pre_id in np.where(pre['spike'] > 0.)[0]:
                syn_ids = pre2syn[pre_id]
                s[syn_ids] += 1
            ST['s'] = s
            ST['g'] = g_max * s


    if mode=='scalar':
        @bp.delayed
        def output(ST, _t, post):
            I_syn = - g_max * ST['s'] * (post['V'] - E)
            post['input'] += I_syn
    elif mode=='vector':
        @bp.delayed
        def output(ST, post, post2syn):
            post_cond = np.zeros(len(post2syn), dtype=np.float_)
            for post_id, syn_ids in enumerate(post2syn):
                post_cond[post_id] = np.sum(ST['g'][syn_ids])
            post['input'] -= post_cond * (post['V'] - E)

    if mode == 'scalar':
        return bp.SynType(name='GABAa_synapse',
                          ST=ST_scalar,
                          requires=requires_scalar,
                          steps=(update, output),
                          mode=mode)
    elif mode == 'vector':
        return bp.SynType(name='GABAa_synapse',
                          ST=ST_vector,
                          requires=requires_vector,
                          steps=(update, output),
                          mode=mode)
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))


def get_GABAa2(g_max=0.04, E=-80., alpha=0.53, beta=0.18, T=1., T_duration=1., mode='vector'):
    """
    GABAa conductance-based synapse model (markov form).

    .. math::
        
        I_{syn}&= - \\bar{g}_{max} s (V - E)

        \\frac{d r}{d t}&=\\alpha[T]^2(1-s) - \\beta s
        
    ST refers to synapse state, members of ST are listed below:
    
    ================ ================= =========================================================
    **Member name**  **Initial Value** **Explanation**
    ---------------- ----------------- ---------------------------------------------------------
    s                0.                Gating variable.
     
    g                0.                Synapse conductance on post-synaptic neuron.
                             
    t_last_pre_spike -1e7              Last spike time stamp of pre-synaptic neuron.
    ================ ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        g_max (float): Maximum synapse conductance.
        E (float): Reversal potential of synapse.
        alpha (float): Opening rate constant of ion channel.
        beta (float): Closing rate constant of ion channel.
        T (float): Transmitter concentration when synapse is triggered by a pre-synaptic spike.
        T_duration (float): Transmitter concentration duration time after being triggered.
        mode (str): Data structure of ST members.

    Returns:
        bp.SynType: return description of GABAa synapse model.

    References:
        .. [1] Destexhe, Alain, and Denis Paré. "Impact of network activity
               on the integrative properties of neocortical pyramidal neurons
               in vivo." Journal of neurophysiology 81.4 (1999): 1531-1547.
    """
    
    
    ST=bp.types.SynState({'s': 0., 'g': 0., 't_last_pre_spike': -1e7}, help = "GABAa synapse state")
    
    requires = dict(
        pre=bp.types.NeuState(['spike'], help = "Pre-synaptic neuron state must have 'spike' item"), 
        post=bp.types.NeuState(['V', 'input'], help = "Post-synaptic neuron state must have 'V' and 'input' item"),
        pre2syn=bp.types.ListConn(help = "Pre-synaptic neuron index -> synapse index"),
        post2syn=bp.types.ListConn(help = "Post-synaptic neuron index -> synapse index")
    )

    @bp.integrate
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    def update(ST, pre, pre2syn, _t):
        for pre_id in np.where(pre['spike'] > 0.)[0]:
            syn_ids = pre2syn[pre_id]
            ST['t_last_pre_spike'][syn_ids] = _t
        TT = ((_t - ST['t_last_pre_spike']) < T_duration) * T
        s = int_s(ST['s'], _t, TT)
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['input'] -= post_cond * (post['V'] - E)

    if mode == 'scalar':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    elif mode == 'vector':
        return bp.SynType(name='GABAa_synapse',
                          ST=ST,
                          requires=requires,
                          steps=[update, output],
                          mode='vector')
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
