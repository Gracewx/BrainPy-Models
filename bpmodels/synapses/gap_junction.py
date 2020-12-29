# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np
import sys

def get_gap_junction(mode='scalar'):
    """
    synapse with gap junction.

    .. math::

        I_{syn} = w (V_{pre} - V_{post})

    ST refers to synapse state, members of ST are listed below:

    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    w                0.                Synapse weights.
    =============== ================= =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        mode (string): data structure of ST members.

    Returns:
        bp.SynType

    Reference:
        .. [1] Chow, Carson C., and Nancy Kopell. 
                "Dynamics of spiking neurons with electrical coupling." 
                Neural computation 12.7 (2000): 1643-1678.

    """

    ST=bp.types.SynState(['w'])

    requires = dict(
        pre=bp.types.NeuState(['V']),
        post=bp.types.NeuState(['V', 'input'])
    )

    if mode=='scalar':
        def update(ST, pre, post):
            post['input'] += ST['w'] * (pre['V'] - post['V'])        

    elif mode == 'vector':
        requires['post2pre']=bp.types.ListConn(help='post-to-pre connection.')
        requires['pre_ids']=bp.types.Array(dim=1, help='Pre-synaptic neuron indices.')

        def update(ST, pre, post, post2pre, pre_ids):
            num_post = len(post2pre)
            for post_id in range(num_post):
                pre_id = pre_ids[post_id]
                post['input'][post_id] += ST['w'] * np.sum(pre['V'][pre_id] - post['V'][post_id])

    elif mode == 'matrix':
        requires['conn_mat']=bp.types.MatConn()

        def update(ST, pre, post, conn_mat):
            # reshape
            dim = np.shape(ST['w'])
            v_post = np.vstack((post['V'],)*dim[0])
            v_pre = np.vstack((pre['V'],)*dim[1]).T   

            # update         
            post['input'] += ST['w'] * (v_pre - v_post) * conn_mat        

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='gap_junction_synapse', 
                        ST=ST, requires=requires, 
                        steps=update, 
                        mode=mode)


def get_gap_junction_lif(weight, k_spikelet=0.1, post_has_refractory=False, mode='scalar'):
    """
    synapse with gap junction.

    .. math::

        I_{syn} = w (V_{pre} - V_{post})

    ST refers to synapse state, members of ST are listed below:

    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    w                0.                Synapse weights.
    
    spikelet         0.                conductance for post-synaptic neuron
    =============== ================= =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        weight (float): Synapse weights.

    Returns:
        bp.SynType

    References:
        .. [1] Chow, Carson C., and Nancy Kopell. 
                "Dynamics of spiking neurons with electrical coupling." 
                Neural computation 12.7 (2000): 1643-1678.

    """

    ST=bp.types.SynState('w', 'spikelet')

    requires = dict(
        pre=bp.types.NeuState(['V', 'spike']),
        post=bp.types.NeuState(['V', 'input'])
    )

    if mode == 'scalar':
        def update(ST, pre, post):
            # gap junction sub-threshold
            post['input'] += ST['w'] * (pre['V'] - post['V'])
            # gap junction supra-threshold
            ST['spikelet'] = ST['w'] * k_spikelet * pre['spike']

        @bp.delayed
        def output(ST, post):
            post['V'] += ST['spikelet']

        steps=(update, output)

    elif mode == 'vector':
        requires['pre2post']=bp.types.ListConn(help='post-to-synapse connection.'),
        requires['pre_ids']=bp.types.Array(dim=1, help='Pre-synaptic neuron indices.'),

        if post_has_refractory:
            requires['post'] = bp.types.NeuState(['V', 'input', 'refractory'])

            def update(ST, pre, post, pre2post):
                num_pre = len(pre2post)
                g_post = np.zeros_like(post['V'], dtype=np.float_)
                spikelet = np.zeros_like(post['V'], dtype=np.float_)
                for pre_id in range(num_pre):
                    post_ids = pre2post[pre_id]
                    pre_V = pre['V'][pre_id]
                    g_post[post_ids] = weight * np.sum(pre_V - post['V'][post_ids])
                    if pre['spike'][pre_id] > 0.:
                        spikelet[post_ids] += weight * k_spikelet * pre_V
                post['V'] += spikelet * (1. - post['refractory'])
                post['input'] += g_post

        else:
            requires['post'] = bp.types.NeuState(['V', 'input'])

            def update(ST, pre, post, pre2post):
                num_pre = len(pre2post)
                g_post = np.zeros_like(post['V'], dtype=np.float_)
                spikelet = np.zeros_like(post['V'], dtype=np.float_)
                for pre_id in range(num_pre):
                    post_ids = pre2post[pre_id]
                    pre_V = pre['V'][pre_id]
                    g_post[post_ids] = weight * np.sum(pre_V - post['V'][post_ids])
                    if pre['spike'][pre_id] > 0.:
                        spikelet[post_ids] += weight * k_spikelet * pre_V
                post['V'] += spikelet
                post['input'] += g_post

        steps=update

    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))

    return bp.SynType(name='gap_junctin_synapse_for_LIF',
                      ST=ST, requires=requires,
                      steps=steps,
                      mode=mode)
