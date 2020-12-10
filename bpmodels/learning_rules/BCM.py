import brainpy as bp
from brainpy import numpy as np
import matplotlib.pyplot as plt
import pdb


def get_BCM(learning_rate=0.01, w_max=2., r_0 = 0.):
    """
    Bienenstock-Cooper-Munro (BCM) rule. (scalar based)

    .. math::

        r_i = \\sum_j w_{ij} r_j 

        \\frac d{dt} w_{ij} = \\eta \\cdot r_i (r_i - r_{\\theta}) r_j

    where :math:`\\eta` is some learning rate, and :math:`r_{\\theta}` is the 
    plasticity threshold,
    which is a function of the averaged postsynaptic rate, we take:

    .. math::

        r_{\\theta} = < r_i >

    ST refers to synapse state (note that BCM learning rule can be implemented as synapses),
    members of ST are listed below:

    ================ ================= =========================================================
    **Member name**  **Initial Value** **Explanation**
    ---------------- ----------------- ---------------------------------------------------------                                  
    w                1.                Synapse weights.
    r_th             0.                Plasticity threshold.
    ================ ================= =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        learning_rate (float): learning rate of the synapse weights.
        w_max (float): Maximum of the synapse weights.
        r_0 (float): Minimal plasticity threshold.

    Returns:
        bp.Syntype: return description of BCM rule.

    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """

    requires = dict(
        ST=bp.types.SynState(
            {'w': 1., 'dwdt': 0., 'r_th': 0., 'post_r': 0., 'sum_r_post':0.}, 
            help='BCM synapse state.'),
        pre=bp.types.NeuState(
            ['r'], help='Pre-synaptic neuron state must have "spike" item.'),
        post=bp.types.NeuState(
            ['r'], help='Post-synaptic neuron state must have "spike" item.'),
        post2syn=bp.types.ListConn(),
        post2pre=bp.types.ListConn()
    )

    def bound(w):
        w = np.where(0 < w, w, 0.001)
        w = np.where(w < w_max, w, w_max)
        return w

    @bp.integrate
    def int_w(w, t, r_pre, r_post, r_th):
        dwdt = learning_rate * r_post * (r_post - r_th) * r_pre
        return (dwdt,),dwdt

    def learn(ST, _t_, pre, post, post2syn, post2pre):
        for i , r_i_post in enumerate(post['r']):
            if r_i_post < r_0:
                ST['post_r'][i] = r_i_post
            else:
                # mapping
                ij = post2syn[i]
                j = post2pre[i]
                r_j_pre = pre['r'][j]
                w_ij = ST['w'][ij]

                # output
                r_i_post = np.dot(w_ij, r_j_pre)

                # threshold
                ST['sum_r_post'][i] += r_i_post
                r_th = ST['sum_r_post'][i] / (_t_ / bp.profile.get_dt() + 1)
                
                # BCM rule
                w_ij, dw_ij = int_w(w_ij, _t_, r_j_pre, r_i_post, r_th)
                w_ij = bound(w_ij)
                ST['w'][ij] = w_ij
                ST['dwdt'][ij] = dw_ij
                ST['post_r'][i] = np.sum(r_i_post)
                ST['r_th'][i] = r_th
            

    @bp.delayed
    def output(ST, post):
        post['r'] = ST['post_r'][:len(post['r'])]

    return bp.SynType(name='BCM_synapse',
                      requires=requires,
                      steps=[learn, output],
                      mode='vector')

