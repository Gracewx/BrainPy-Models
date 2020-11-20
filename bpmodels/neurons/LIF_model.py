# -*- coding: utf-8 -*-

import brainpy as bp


def get_LIF(V_rest=0., V_reset=-5., V_th=20., R=1.,
            tau=10., t_refractory=5., noise=0.):
    """Leaky Integrate-and-Fire neuron model.
        
    .. math::

        \\tau \\frac{d V}{d t}&=-(V-V_{rest}) + RI(t)
    
    Args:
        V_rest (float): Resting potential.
        V_reset (float): Reset potential after spike.
        V_th (float): Threshold potential of spike.
        R (float): Membrane resistance.
        C (float): Membrane capacitance.
        tau (float): Membrane time constant. Compute by R * C.
        t_refractory (int): Refractory period length.(ms)
        noise (float): noise.   
        
    Returns:
        bp.Neutype: return description of LIF model.
    """

    ST = bp.types.NeuState(
        {'V': 0, 'input': 0, 'spike': 0, 'refractory': 0, 't_last_spike': -1e7}
    )

    @bp.integrate
    def int_V(V, _t_, I_ext):  # integrate u(t)
        return (- (V - V_rest) + R * I_ext) / tau, noise / tau

    def update(ST, _t_):
        # update variables
        ST['spike'] = 0
        if _t_ - ST['t_last_spike'] <= t_refractory:
            ST['refractory'] = 1.
        else:
            ST['refractory'] = 0.
            V = int_V(ST['V'], _t_, ST['input'])
            if V >= V_th:
                V = V_reset
                ST['spike'] = 1
                ST['t_last_spike'] = _t_
            ST['V'] = V
    
    def reset(ST):
        ST['input'] = 0.

    return bp.NeuType(name='LIF_neuron',
                      requires=dict(ST=ST),
                      steps=(update, reset),
                      vector_based=False)
