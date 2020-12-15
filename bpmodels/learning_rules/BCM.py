import brainpy as bp
from brainpy import numpy as np
import matplotlib.pyplot as plt
import sys

def get_BCM(learning_rate=0.01, w_max=2., w_min = 0., r_0 = 0., mode='matrix'):
    """
    Bienenstock-Cooper-Munro (BCM) rule.

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
    ================ ================= =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        learning_rate (float): learning rate of the synapse weights.
        w_max (float): Maximum of the synapse weights.
        w_min (float): Minimum of the synapse weights.
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
            {'w': 1., 'dwdt': 0.}, 
            help='BCM synapse state.'),
        pre=bp.types.NeuState(
            ['r'], help='Pre-synaptic neuron state must have "spike" item.'),
        post=bp.types.NeuState(
            ['r'], help='Post-synaptic neuron state must have "spike" item.'),
        r_th = bp.types.Array(1),
        post_r = bp.types.Array(1),
        sum_r_post = bp.types.Array(1)        
    )

    def bound(w):
        w = np.where(0 < w, w, w_min)
        w = np.where(w < w_max, w, w_max)
        return w

    @bp.integrate
    def int_w(w, t, r_pre, r_post, r_th):
        dwdt = learning_rate * r_post * (r_post - r_th) * r_pre
        return (dwdt,),dwdt

    if mode == 'scalar':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))

    elif mode == 'vector':
        requires['post2syn']=bp.types.ListConn()
        requires['post2pre']=bp.types.ListConn()

        def learn(ST, _t_, pre, post, post2syn, post2pre, r_th, sum_r_post, post_r):
            for i , r_i_post in enumerate(post['r']):
                if r_i_post < r_0:
                    post_r[i] = r_i_post
                elif post2syn[i].size > 0 and post2pre[i].size > 0:
                    # mapping
                    ij = post2syn[i]
                    j = post2pre[i]
                    r_j_pre = pre['r'][j]
                    w_ij = ST['w'][ij]

                    # threshold
                    sum_r_post[i] += r_i_post
                    r_threshold = sum_r_post[i] / (_t_ / bp.profile._dt + 1)
                    r_th[i] = r_threshold
                    
                    # BCM rule
                    w_ij, dw_ij = int_w(w_ij, _t_, r_j_pre, r_i_post, r_th[i])
                    w_ij = bound(w_ij)
                    ST['w'][ij] = w_ij
                    ST['dwdt'][ij] = dw_ij

                    # output
                    next_post_r = np.dot(w_ij, r_j_pre)
                    post_r[i] = next_post_r
                             
        @bp.delayed
        def output(post, post_r):
            post['r'] = post_r

    elif mode == 'matrix':
        requires['conn_mat']=bp.types.MatConn()

        def learn(ST, _t_, pre, post, conn_mat, r_th, sum_r_post, post_r):
            r_i_post = post['r']
            w_ij = ST['w'] * conn_mat
            r_j_pre = pre['r']

            # threshold
            sum_r_post += r_i_post
            r_th = sum_r_post / (_t_ / bp.profile._dt + 1)          

            # BCM rule
            dim = np.shape(w_ij)
            reshape_th = np.vstack((r_th,)*dim[0])
            reshape_post = np.vstack((r_i_post,)*dim[0])
            reshape_pre = np.vstack((r_j_pre,)*dim[1]).T
            w_ij, dw_ij = int_w(w_ij, _t_, reshape_pre, reshape_post, reshape_th)
            w_ij = bound(w_ij)
            ST['w'] = w_ij
            ST['dwdt'] = dw_ij

            # output
            next_post_r = np.dot(w_ij.T, r_j_pre)
            post_r = next_post_r

        @bp.delayed
        def output(post, post_r):
            post['r'] = post_r


    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='BCM_synapse',
                      requires=requires,
                      steps=[learn, output],
                      mode=mode)