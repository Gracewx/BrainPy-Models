import brainpy as bp
import numpy as np
import sys

def get_Oja(gamma = 0.005, w_max = 1., w_min = 0., mode = 'vector'):
    """
    Oja's learning rule.

    .. math::
        
        \\frac{d w_{ij}}{dt} = \\gamma(\\upsilon_i \\upsilon_j - w_{ij}\\upsilon_i ^ 2)
    
    ST refers to synapse state (note that Oja learning rule can be implemented as synapses),
    members of ST are listed below:
    
    ================ ================= =========================================================
    **Member name**  **Initial Value** **Explanation**
    ---------------- ----------------- ---------------------------------------------------------
    w                0.05              Synapse weight.

    output_save      0.                Temporary save synapse output value until post-synaptic
                                       neuron get the value after delay time.
    ================ ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).
    
    Args:
        gamma(float): Learning rate.
        w_max (float): Maximal possible synapse weight.
        w_min (float): Minimal possible synapse weight.
        mode (str): Data structure of ST members.
        
    Returns:
        bp.Syntype: return description of synapse with Oja's rule.
        
    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """
    requires = dict(
        ST = bp.types.SynState({'w': 0.05, 'output_save': 0.}),
        pre = bp.types.NeuState(['r']),
        post = bp.types.NeuState(['r']), 
        post2syn=bp.types.ListConn(),
        post2pre=bp.types.ListConn(),
    )
    
    @bp.integrate
    def int_w(w, _t_, r_pre, r_post):
        dw = gamma * (r_post * r_pre - np.square(r_post) * w)
        return dw

    def update(ST, _t_, pre, post, post2syn, post2pre):
        for post_id, post_r in enumerate(post['r']):
            syn_ids = post2syn[post_id]
            pre_ids = post2pre[post_id]
            pre_r = pre['r'][pre_ids]
            w = ST['w'][syn_ids]
            output = np.dot(w, pre_r)
            output += post_r
            w = int_w(w, _t_, pre_r, output)
            ST['w'][syn_ids] = w
            ST['output_save'][syn_ids] = output
    
    @bp.delayed
    def output(ST, pre, post, post2syn):
        for post_id, _ in enumerate(post['r']):
            syn_ids = post2syn[post_id]
            post['r'][post_id] += ST['output_save'][syn_ids[0]]
    
    if mode == 'scalar':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    elif mode == 'vector':
        return bp.SynType(name='Oja_synapse',
                          requires=requires,
                          steps=(update, output),
                          mode=mode)
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
