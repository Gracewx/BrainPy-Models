# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.numpy as np

def get_GABAb1(g_max=0.02, E=-95., k1=0.18, k2=0.034, k3=0.09, k4=0.0012, 
               kd=100., T=0.5, T_duration=0.3):
    """GABAb conductance-based synapse model(type 1).

    .. math::

        &\\frac{d[R]}{dt} = k_3 [T](1-[R])- k_4 [R]

        &\\frac{d[G]}{dt} = k_1 [R]- k_2 [G]

        I_{GABA_{B}} &=\\overline{g}_{GABA_{B}} (\\frac{[G]^{4}} {[G]^{4}+K_{d}}) (V-E_{GABA_{B}})


    - [G] is the concentration of activated G protein.
    - [R] is the fraction of activated receptor.
    - [T] is the transmitter concentration.

    Args:
        g_max (float): Maximum synapse conductance.
        E (float): Reversal potential of synapse.
        k1 (float): Activating rate constant of G protein catalyzed by activated GABAb receptor.
        k2 (float): De-activating rate constant of G protein.
        k3 (float): Activating rate constant of GABAb receptor.
        k4 (float): De-activating rate constant of GABAb receptor.
        T (float): Transmitter concentration when synapse is triggered by a pre-synaptic spike.
        T_duration (float): Transmitter concentration duration time after being triggered.

    Returns:
        bp.SynType: return description of GABAb synapse model.
    """

    requires = dict(
        ST=bp.types.SynState({'R': 0., 'G': 0., 't_last_pre_spike': -1e7, 'g': 0.}, help = "GABAb synapse state"),
        pre=bp.types.NeuState(['spike'], help = "Pre-synaptic neuron state must have 'spike' item"),
        post=bp.types.NeuState(['V', 'input'], help = "Post-synaptic neuron state must have 'V' and 'input' item"),
        pre2syn=bp.types.ListConn(help = "Pre-synaptic neuron index -> synapse index"),
        post2syn=bp.types.ListConn(help = "Post-synaptic neuron index -> synapse index"),
    )

    @bp.integrate
    def int_R(R, t, TT):
        return k3 * TT * (1 - R) - k4 * R

    @bp.integrate
    def int_G(G, t, R):
        return k1 * R - k2 * G

    def update(ST, _t_, pre, pre2syn):
        for pre_id in np.where(pre['spike'] > 0.)[0]:
            syn_ids = pre2syn[pre_id]
            ST['t_last_pre_spike'][syn_ids] = _t_
        TT = ((_t_ - ST['t_last_pre_spike']) < T_duration) * T
        R = int_R(ST['R'], _t_, TT)
        G = int_G(ST['G'], _t_, R)
        ST['R'] = R
        ST['G'] = G
        ST['g'] = g_max * G ** 4 / (G ** 4 + kd)

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['input'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='GABAb1',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)


def get_GABAb2(g_max=0.02, E=-95., k1=0.66, k2=0.02, k3=0.0053, k4=0.017,
               k5=8.3e-5, k6=7.9e-3, kd=100., T=0.5, T_duration=0.5):
    """
    GABAb conductance-based synapse model (markov form).
    
    G-protein cascade occurs in the following steps: 
    (i) the transmitter binds to the receptor, leading to its activated form; 
    (ii) the activated receptor catalyzes the activation of G proteins; 
    (iii) G proteins bind to open K+ channel, with n(=4) independent binding sites.

    .. math::

        &\\frac{d[D]}{dt}=K_{4}[R]-K_{3}[D]

        &\\frac{d[R]}{dt}=K_{1}[T](1-[R]-[D])-K_{2}[R]+K_{3}[D]

        &\\frac{d[G]}{dt}=K_{5}[R]-K_{6}[G]

        I_{GABA_{B}}&=\\bar{g}_{GABA_{B}} \\frac{[G]^{n}}{[G]^{n}+K_{d}}(V-E_{GABA_{B}})

    - [R] is the fraction of activated receptor.
    - [D] is the fraction of activated receptor.
    - [G] is the concentration of activated G-protein (μM).
    - [T] is the transmitter concentration.

    Args:
        g_max (float): Maximum synapse conductance.
        E (float): Reversal potential of synapse.
        k1 (float): Activating rate constant of GABAb receptor.
        k2 (float): De-activating rate constant of GABAb receptor.
        k3 (float): Activating rate constant of desensitized GABAb receptor.
        k4 (float): Desensitizing rate constant of activated GABAb receptor. 
        k5 (float): Activating rate constant of G protein catalyzed by activated GABAb receptor.
        k6 (float): De-activating rate constant of activated G protein.
        kd (float): Dissociation constant of the binding of G protein on K+ channels.
        T (float): Transmitter concentration when synapse is triggered by a pre-synaptic spike.
        T_duration (float): Transmitter concentration duration time after being triggered.

    Returns:
        bp.SynType: return decription of GABAb synapse model.

    """
    requires = dict(
        ST=bp.types.SynState({'D': 0., 'R': 0., 'G': 0., 't_last_pre_spike': -1e7, 'g': 0.}, help = "GABAb synapse state"),
        pre=bp.types.NeuState(['spike'], help = "Pre-synaptic neuron state must have 'spike' item"),
        post=bp.types.NeuState(['V', 'input'], help = "Post-synaptic neuron state must have 'V' and 'input' item"),
        pre2syn=bp.types.ListConn(help = "Pre-synaptic neuron index -> synapse index"),
        post2syn=bp.types.ListConn(help = "Post-synaptic neuron index -> synapse index"),
    )

    @bp.integrate
    def int_D(D, t, R):
        return k4 * R - k3 * D

    @bp.integrate
    def int_R(R, t, TT, D):
        return k1 * TT * (1 - R - D) - k2 * R + k3 * D

    @bp.integrate
    def int_G(G, t, R):
        return k5 * R - k6 * G

    def update(ST, _t_, pre, pre2syn):
        # calculate synaptic state
        for pre_id in np.where(pre['spike'] > 0.)[0]:
            syn_ids = pre2syn[pre_id]
            ST['t_last_pre_spike'][syn_ids] = _t_
        TT = ((_t_ - ST['t_last_pre_spike']) < T_duration) * T
        D = int_D(ST['D'], _t_, ST['R'])
        R = int_R(ST['R'], _t_, TT, D)
        G = int_G(ST['G'], _t_, R)
        ST['D'] = D
        ST['R'] = R
        ST['G'] = G
        ST['g'] = - g_max * (G ** 4 / (G ** 4 + kd))

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['input'] += post_cond * (post['V'] - E)

    return bp.SynType(name='GABAb2',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)