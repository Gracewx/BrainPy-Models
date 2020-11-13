# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.numpy as np


def get_GABAa1(g_max=0.4, reversal_potential=-80., tau_decay=6.):
    """GABAa synapse model.(scalar)

    .. math::

        I_{syn}&= - \\bar{g}_{syn} s (V-E_{syn})

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})

    Args:
        g_max (float): Maximum conductance.
        E (float): Reversal potential.
        tau_decay (float): Time constant for s decay.

    Returns:
        bp.Syntype: return description of GABAa model.
    """
    requires = dict(
        ST=bp.types.SynState(['s', 'g']),
        pre=bp.types.NeuState(['sp']),
        pre2syn=bp.types.ListConn(),
    )

    @bp.integrate
    def int_s(s, t):
        return - s / tau_decay

    def update(ST, pre, pre2syn):
        s = int_s(ST['s'], 0.)
        for pre_id in np.where(pre['sp'] > 0.)[0]:
            syn_ids = pre2syn[pre_id]
            s[syn_ids] += 1
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['inp'] -= post_cond * (post['V'] - reversal_potential)

    return bp.SynType(name='GABAa1',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)


