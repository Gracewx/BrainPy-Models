# -*- coding: utf-8 -*-
import brainpy as bp
import brainpy.numpy as np

def get_exponential(g_max=0.2, E=-60., tau=8):
    '''
    exponential synapse model.

    .. math::

        I_{syn}(t) &= g_{syn} (t) (V(t)-E_{syn})

        g_{syn} (t) &= \\bar{g}_{syn} exp(- \\frac{t-t_f}{\\tau})

    ST refers to synapse state, members of ST are listed below:
    
    ================ ======== =========================================================
    **Member name**  **Type** **Explanation**
    ---------------- -------- ---------------------------------------------------------     
    g                float    Synapse conductance on post-synaptic neuron.
                             
    t_last_pre_spike float    Last spike time stamp of pre-synaptic neuron.
    ================ ======== =========================================================

    Args:
        g_max (float): The peak conductance change in µmho (µS).
        E (float): The reversal potential for the synaptic current.
        tau (float): The time constant of decay.

    Returns:
        bp.Neutype: return description of exponential synapse model.
    '''

    requires = {
        'ST': bp.types.SynState(['g', 't_last_pre_spike'],help='The conductance defined by exponential function.'),
        'pre': bp.types.NeuState(['spike'], help='pre-synaptic neuron state must have "V"'),
        'post': bp.types.NeuState(['input', 'V'], help='post-synaptic neuron state must include "input" and "V"'),
        'pre2syn': bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index'),
        'post2syn': bp.types.ListConn(help='Post-synaptic neuron index -> synapse index'),
    }


    def update(ST, _t_, pre, pre2syn):
        for pre_idx in np.where(pre['spike'] > 0.)[0]:
            syn_idx = pre2syn[pre_idx]
            ST['t_last_pre_spike'][syn_idx] = _t_
        g = g_max * np.exp(-(_t_-ST['t_last_pre_spike']) / tau)
        ST['g'] = g

    @bp.delayed
    def output(ST, post, post2syn):
        I_syn = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            I_syn[post_id] = np.sum(ST['g'][syn_ids]*(post['V'] - E))
        post['input'] -= I_syn

    return bp.SynType(name='exponential_synapse',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)
