import brainpy as bp
import brainpy.numpy as np

def get_alpha(tau_decay = 2.):

    """alpha conductance-based synapse.

    Parameters
    ----------
    tau_decay : float
        The time constant of decay.
    """


    requires = dict(
        ST = bp.types.SynState(['s', 'g', 'w']),
        pre=bp.types.NeuState(['spike']),
        post=bp.types.NeuState(['V', 'input']),
        pre2syn=bp.types.ListConn(),
        post2syn=bp.types.ListConn(),
    )

    @bp.integrate
    def ints(s, t):
        return - s / tau_decay


    def update(ST, _t_, pre, pre2syn):
        s = ints(ST['s'], _t_)
        for i in range(pre['spike'].shape[0]):
            if pre['spike'][i] > 0.:
                syn_ids = pre2syn[i]
                s[syn_ids] += 1.
        ST['s'] = s
        ST['g'] = ST['w'] * s


    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['input'] += post_cond


    return bp.SynType(name='alpha_synapse',
                 requires=requires,
                 steps=(update, output),
                 vector_based=True)