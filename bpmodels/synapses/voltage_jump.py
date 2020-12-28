import brainpy as bp
import brainpy.numpy as np
import sys

def get_voltage_jump(post_has_refractory=False, mode='vector'):
    """Voltage jump synapses without post-synaptic neuron refractory.

    .. math::

        I_{syn} = \sum J \delta(t-t_j)


    ST refers to synapse state, members of ST are listed below:
    
    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    g               0.                Synapse conductance on post-synaptic neuron.
    =============== ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        post_has_refractory (bool): whether the post-synaptic neuron have refractory.

    Returns:
        bp.SynType.
    
  
    """
    if mode=='vector':

        ST=bp.types.SynState(['g'])

        requires = dict(
            pre=bp.types.NeuState(['spike']),
            pre2post=bp.types.ListConn(),
        )

        if post_has_refractory:
            requires['post'] = bp.types.NeuState(['V', 'refractory'])

            @bp.delayed
            def output(ST, post):
                post['V'] += ST['g'] * (1. - post['refractory'])

        else:
            requires['post'] = bp.types.NeuState(['V'])

            @bp.delayed
            def output(ST, post):
                post['V'] += ST['g']

        def update(ST, pre, post, pre2post):
            num_post = post['V'].shape[0]
            g = np.zeros_like(num_post, dtype=np.float_)
            for pre_id in range(pre['spike'].shape[0]):
                if pre['spike'][pre_id] > 0.:
                    post_ids = pre2post[pre_id]
                    g[post_ids] = 1.
            ST['g'] = g

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='voltage_jump_synapse',
                      ST=ST, requires=requires,
                      steps=(update, output),
                      mode = mode)
