import brainpy as bp
import numpy as np


def get_alpha(g_max=.2, E=0., tau_decay = 2.):

    """
    Alpha conductance-based synapse.

    .. math::
    
        I_{syn}(t) &= g_{syn} (t) (V(t)-E_{syn})

        g_{syn} (t) &= \\sum \\bar{g}_{syn} \\frac{t-t_f} {\\tau} exp(- \\frac{t-t_f}{\\tau})  

    ST refers to the synapse state, items in ST are listed below:
    
    ================ ================== =========================================================
    **Member name**  **Initial values** **Explanation**
    ---------------- ------------------ ---------------------------------------------------------    
    g                  0                  Synapse conductance on the post-synaptic neuron.
                             
    t_last_pre_spike   -1e7               Last spike time stamp of the pre-synaptic neuron.
    ================ ================== =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        g_max (float): The peak conductance change in µmho (µS).
        E (float): The reversal potential for the synaptic current.
        tau_decay (float): The time constant of decay.

    Returns:
        bp.Neutype

    References:
        .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw. 
                "The Synapse." Principles of Computational Modelling in Neuroscience. 
                Cambridge: Cambridge UP, 2011. 172-95. Print.
    """

    ST=bp.types.SynState({'g': 0., 't_last_pre_spike': -1e7}, help='The conductance defined by exponential function.')

    requires = {
        'pre': bp.types.NeuState(['spike'], help='pre-synaptic neuron state must have "V"'),
        'post': bp.types.NeuState(['input', 'V'], help='post-synaptic neuron state must include "input" and "V"'),
        'pre2syn': bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index'),
        'post2syn': bp.types.ListConn(help='Post-synaptic neuron index -> synapse index'),
    }

    def update(ST, _t, pre, pre2syn):
        for pre_idx in np.where(pre['spike'] > 0.)[0]:
            syn_idx = pre2syn[pre_idx]
            ST['t_last_pre_spike'][syn_idx] = _t
        c = (_t-ST['t_last_pre_spike']) / tau_decay
        g = g_max * np.exp(-c) * c
        ST['g'] = g

    @bp.delayed
    def output(ST, post, post2syn):
        I_syn = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            I_syn[post_id] = np.sum(ST['g'][syn_ids]*(post['V'] - E))
        post['input'] -= I_syn

    return bp.SynType(name='alpha_synapse',
                 requires=requires,
                 ST=ST,
                 steps=(update, output),
                 mode = 'vector')



def get_alpha2(g_max=.2, E=0., tau_decay = 2.):

    """
    Alpha conductance-based synapse. 

    .. math::
    
        I_{syn}(t) &= g_{syn} (t) (V(t)-E_{syn})

        g_{syn} (t) &= w s

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})


    ST refers to the synapse state, items in ST are listed below:
    
    ================ ================== =========================================================
    **Member name**  **Initial values** **Explanation**
    ---------------- ------------------ ---------------------------------------------------------    
    g                  0                  Synapse conductance on the post-synaptic neuron.
    s                  0                  Synapse conductance on the post-synaptic neuron.
    w                  1                  Synapse conductance on the post-synaptic neuron.  
    ================ ================== =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        g_max (float): The peak conductance change in µmho (µS).
        E (float): The reversal potential for the synaptic current.
        tau_decay (float): The time constant of decay.

    Returns:
        bp.Neutype
    """

    ST=bp.types.SynState({'g': 0., 's': 0., 'w':1.}, help='The conductance defined by exponential function.')

    requires = {
        'pre': bp.types.NeuState(['spike'], help='pre-synaptic neuron state must have "V"'),
        'post': bp.types.NeuState(['input', 'V'], help='post-synaptic neuron state must include "input" and "V"'),
        'pre2syn': bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index'),
        'post2syn': bp.types.ListConn(help='Post-synaptic neuron index -> synapse index'),
    }


    @bp.integrate
    def ints(s, t):
        return - s / tau_decay


    def update(ST, _t, pre, pre2syn):
        s = ints(ST['s'], _t)
        for i in np.where(pre['spike'] > 0.)[0]:
            syn_ids = pre2syn[i]
            s[syn_ids] += 1.
        ST['s'] = s
        ST['g'] = ST['w'] * s


    def output(ST, post, post2syn):
        for post_id, syn_id in enumerate(post2syn):
            post['input'][post_id] += np.sum(ST['g'][syn_id])

    return bp.SynType(name='alpha_synapse',
                 requires=requires,
                 ST=ST,
                 steps=(update, output),
                 mode = 'vector')