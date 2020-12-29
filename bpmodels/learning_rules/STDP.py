# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np
import sys

def get_STDP1(g_max=0.10, E=0., tau_decay=10., tau_s = 10., tau_t = 10., 
              w_min = 0., w_max = 20., delta_A_s = 0.5, delta_A_t = 0.5, mode='vector'):
    """
    Spike-time dependent plasticity (in differential form).
    
    .. math::

        \\frac{d A_{source}}{d t}&=-\\frac{A_{source}}{\\tau_{source}}
        
        \\frac{d A_{target}}{d t}&=-\\frac{A_{target}}{\\tau_{target}}
    
    After a pre-synaptic spike:

    .. math::      
      
        g_{post}&= g_{post}+w
        
        A_{source}&= A_{source} + \\delta A_{source}
        
        w&= min([w-A_{target}]^+, w_{max})
        
    After a post-synaptic spike:
    
    .. math::
        
        A_{target}&= A_{target} + \\delta A_{target}
        
        w&= min([w+A_{source}]^+, w_{max})

    ST refers to synapse state (note that STDP learning rule can be implemented as synapses),
    members of ST are listed below:
    
    ================ ================= =========================================================
    **Member name**  **Initial Value** **Explanation**
    ---------------- ----------------- ---------------------------------------------------------
    A_s              0.                Source neuron trace.
    
    A_t              0.                Target neuron trace.
     
    g                0.                Synapse conductance on post-synaptic neuron.
                             
    w                0.                Synapse weight.
    ================ ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).
    
    Args:
        g_max (float): Maximum conductance.
        E (float): Reversal potential.
        tau_decay (float): Time constant of decay.
        tau_s (float): Time constant of source neuron (i.e. pre-synaptic neuron)
        tau_t (float): Time constant of target neuron (i.e. post-synaptic neuron)
        w_min (float): Minimal possible synapse weight.
        w_max (float): Maximal possible synapse weight.
        delta_A_s (float): Change on source neuron traces elicited by a source neuron spike.
        delta_A_t (float): Change on target neuron traces elicited by a target neuron spike.
        mode (str): Data structure of ST members.
        
    Returns:
        bp.Syntype: return description of STDP.
        
    References:
        .. [1] Stimberg, Marcel, et al. "Equation-oriented specification of neural models for
               simulations." Frontiers in neuroinformatics 8 (2014): 6.
    """
    requires_scalar = dict(
        ST=bp.types.SynState(['A_s', 'A_t', 'g', 'w'], 
                             help='STDP synapse state.'),
        pre=bp.types.NeuState(['spike'], help='Pre-synaptic neuron state \
                                               must have "spike" item.'),
        post=bp.types.NeuState(['V', 'input', 'spike'], 
                               help='Pre-synaptic neuron state must \
                                     have "V", "input" and "spike" item.'),
    )

    requires_vector = dict(
        ST=bp.types.SynState({'A_s': 0., 'A_t': 0., 'g': 0., 'w': 0.}, 
                             help='STDP synapse state.'),
        pre=bp.types.NeuState(['spike'], help='Pre-synaptic neuron state \
                                               must have "spike" item.'),
        post=bp.types.NeuState(['V', 'input', 'spike'], 
                               help='Post-synaptic neuron state must \
                                     have "V", "input" and "spike" item.'),
        pre2syn=bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index'),
        post2syn=bp.types.ListConn(help='Post-synaptic neuron index -> synapse index'),
    )

    @bp.integrate
    def int_A_s(A_s, _t_):
        return -A_s / tau_s

    @bp.integrate
    def int_A_t(A_t, _t_):
        return -A_t / tau_t

    @bp.integrate
    def int_g(g, _t_):
        return -g / tau_decay

    if mode=='scalar':
        def my_relu(w):
            return w if w > 0 else 0
    
    if mode=='scalar':
        def update(ST, _t_, pre, post):
            A_s = int_A_s(ST['A_s'], _t_)
            A_t = int_A_t(ST['A_t'], _t_)
            g = int_g(ST['g'], _t_)
            w = ST['w']
            if pre['spike']:
                g += ST['w']
                A_s = A_s + delta_A_s
                w = np.clip(my_relu(ST['w'] - A_t), w_min, w_max)
            if post['spike']:
                A_t = A_t + delta_A_t
                w = np.clip(my_relu(ST['w'] + A_s), w_min, w_max)
            ST['A_s'] = A_s
            ST['A_t'] = A_t
            ST['g'] = g
            ST['w'] = w
    elif mode=='vector':
        def update(ST, _t_, pre, post, pre2syn, post2syn):
            A_s = int_A_s(ST['A_s'], _t_)
            A_t = int_A_t(ST['A_t'], _t_)
            g = int_g(ST['g'], _t_)
            w = ST['w']
            for i in np.where(pre['spike'] > 0.)[0]:
                syn_ids = pre2syn[i]
                g[syn_ids] += ST['w'][syn_ids]
                A_s[syn_ids] = A_s[syn_ids] + delta_A_s
                w[syn_ids] = np.clip(ST['w'][syn_ids] - ST['A_t'][syn_ids], w_min, w_max)
            for i in np.where(post['spike'] > 0.)[0]:
                syn_ids = post2syn[i]
                A_t[syn_ids] = A_t[syn_ids] + delta_A_t
                w[syn_ids] = np.clip(ST['w'][syn_ids] + ST['A_s'][syn_ids], w_min, w_max)
            ST['A_s'] = A_s
            ST['A_t'] = A_t
            ST['g'] = g
            ST['w'] = w

    if mode=='scalar':
        @bp.delayed
        def output(ST, post):
            I_syn = - g_max * ST['g'] * (post['V'] - E)
            post['input'] += I_syn
    elif mode=='vector':
        @bp.delayed
        def output(ST, post, post2syn):
            post_cond = np.zeros(len(post2syn), dtype=np.float_)
            for post_id, syn_ids in enumerate(post2syn):
                post_cond[post_id] = np.sum(- g_max * ST['g'][syn_ids] * (post['V'][post_id] - E))
            post['input'] += post_cond

    if mode == 'scalar':
        return bp.SynType(name='STDP_synapse',
                          requires=requires_scalar,
                          steps=(update, output),
                          mode=mode)
    elif mode == 'vector':
        return bp.SynType(name='STDP_synapse',
                          requires=requires_vector,
                          steps=(update, output),
                          mode=mode)
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))


def get_STDP2(g_max=0.10, E=0., tau_decay=10., tau_s = 10., tau_t = 10., 
              w_min = 0., w_max = 20., delta_A_s = 0.5, delta_A_t = 0.5, mode='vector'):
    """
    Spike-time dependent plasticity (in integrated form).
    
    After a pre-synaptic spike:

    .. math::      
      
        g_{post}&= g_{post}+w
        
        A_{source}&= A_{source}*e^{\\frac{last update-t}{\\tau_{source}}} + \\delta A_{source}
        
        A_{target}&= A_{target}*e^{\\frac{last update-t}{\\tau_{target}}}
        
        w&= min([w-A_{target}]^+, w_{max})
        
    After a post-synaptic spike:
    
    .. math::
        
        A_{source}&= A_{source}*e^{\\frac{last update-t}{\\tau_{source}}}
        
        A_{target}&= A_{target}*e^{\\frac{last update-t}{\\tau_{target}}} + \\delta A_{target}
        
        w&= min([w+A_{source}]^+, w_{max})

    ST refers to synapse state (note that STDP learning rule can be implemented as synapses),
    members of ST are listed below:
    
    ================ ================= =========================================================
    **Member name**  **Initial Value** **Explanation**
    ---------------- ----------------- ---------------------------------------------------------
    A_s              0.                Source neuron trace.
    
    A_t              0.                Target neuron trace.
     
    g                0.                Synapse conductance on post-synaptic neuron.
    
    w                0.                Synapse weight.
                             
    last_spike       -1e7              Last spike time stamp of pre or post-synaptic neuron.
    ================ ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).
    
    Args:
        g_max (float): Maximum conductance.
        E (float): Reversal potential.
        tau_decay (float): Time constant of decay.
        tau_s (float): Time constant of source neuron (i.e. pre-synaptic neuron)
        tau_t (float): Time constant of target neuron (i.e. post-synaptic neuron)
        w_min (float): Minimal possible synapse weight.
        w_max (float): Maximal possible synapse weight.
        delta_A_s (float): Change on source neuron traces elicited by a source neuron spike.
        delta_A_t (float): Change on target neuron traces elicited by a target neuron spike.
        mode (str): Data structure of ST members.
        
    Returns:
        bp.Syntype: return description of STDP.
        
    References:
        .. [1] Stimberg, Marcel, et al. "Equation-oriented specification of neural models for
               simulations." Frontiers in neuroinformatics 8 (2014): 6.
    """

    requires_scalar = dict(
        ST=bp.types.SynState(['A_s', 'A_t', 'g', 'w', 'last_spike'], help='AMPA synapse state.'),
        pre=bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "spike" item.'),
        post=bp.types.NeuState(['V', 'input', 'spike'], help='Pre-synaptic neuron state must have "V" and "input" item.'),
    )

    requires_vector = dict(
        ST=bp.types.SynState({'A_s': 0., 'A_t': 0., 'g': 0., 'w': 0., 'last_spike':-1e7}, help='STDP synapse state.'),
        pre=bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "spike" item.'),
        post=bp.types.NeuState(['V', 'input', 'spike'], help='Post-synaptic neuron state must have "V", "input" and "spike" item.'),
        pre2syn=bp.types.ListConn(
            help='Pre-synaptic neuron index -> synapse index'),
        post2syn=bp.types.ListConn(
            help='Post-synaptic neuron index -> synapse index'),
    )

    @bp.integrate
    def int_g(g, _t_):
        return -g / tau_decay
    
    if mode=='scalar':
        def my_relu(w):
            return w if w > 0 else 0

    if mode=='scalar':
        def update(ST, _t_, pre, post):
            g = int_g(ST['g'], _t_)
            w = ST['w']
            if pre['spike']:
                g += w
                ST['A_s'] = ST['A_s'] * np.exp((ST['last_spike'] - _t_) / tau_s) + delta_A_s
                ST['A_t'] = ST['A_t'] * np.exp((ST['last_spike'] - _t_) / tau_t)
                w = np.clip(my_relu(ST['w'] - ST['A_t']), w_min, w_max)
                ST['last_spike'] = _t_
            if post['spike']:
                ST['A_s'] = ST['A_s'] * np.exp((ST['last_spike'] - _t_) / tau_s)
                ST['A_t'] = ST['A_t'] * np.exp((ST['last_spike'] - _t_) / tau_t) + delta_A_t
                w = np.clip(my_relu(ST['w'] + ST['A_s']), w_min, w_max)
                ST['last_spike']  =_t_
            ST['w'] = w
            ST['g'] = g
    elif mode=='vector':
        def update(ST, _t_, pre, post, pre2syn, post2syn):
            g = int_g(ST['g'], _t_)
            w = ST['w']
            for i in np.where(pre['spike'] > 0.)[0]:
                syn_id = pre2syn[i]
                g[syn_id] += ST['w'][syn_id]
                ST['A_s'][syn_id] = ST['A_s'][syn_id] * np.exp((ST['last_spike'][syn_id] - _t_) / tau_s) + delta_A_s
                ST['A_t'][syn_id] = ST['A_t'][syn_id] * np.exp((ST['last_spike'][syn_id] - _t_) / tau_t)
                w[syn_id] = np.clip(ST['w'][syn_id] - ST['A_t'][syn_id], w_min, w_max)
                ST['last_spike'][syn_id] = _t_
            for i in np.where(post['spike'] > 0.)[0]:
                syn_id = post2syn[i]
                ST['A_s'][syn_id] = ST['A_s'][syn_id] * np.exp((ST['last_spike'][syn_id] - _t_) / tau_s)
                ST['A_t'][syn_id] = ST['A_t'][syn_id] * np.exp((ST['last_spike'][syn_id] - _t_) / tau_t) + delta_A_t
                w[syn_id] = np.clip(ST['w'][syn_id] + ST['A_s'][syn_id], w_min, w_max)
                ST['last_spike'][syn_id]  =_t_
            ST['w'] = w
            ST['g'] = g

    if mode=='scalar':
        @bp.delayed
        def output(ST, post):
            I_syn = - g_max * ST['w'] * (post['V'] - E)
            post['input'] += I_syn
    elif mode=='vector':
        @bp.delayed
        def output(ST, post, post2syn):
            post_cond = np.zeros(len(post2syn), dtype=np.float_)
            for post_id, syn_ids in enumerate(post2syn):
                post_cond[post_id] = np.sum(- g_max * ST['g'][syn_ids] * (post['V'][post_id] - E))
            post['input'] += post_cond

    if mode == 'scalar':
        return bp.SynType(name='STDP_synapse',
                          requires=requires_scalar,
                          steps=(update, output),
                          mode='scalar')
    elif mode == 'vector':
        return bp.SynType(name='STDP_synapse',
                          requires=requires_vector,
                          steps=(update, output),
                          mode=mode)
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
