# -*- coding: utf-8 -*-
import brainpy as bp
import brainpy.numpy as np


def get_MorrisLecar(noise=0., V_Ca=120., g_Ca=4.4, V_K=-84., g_K=8., V_Leak=-60.,
                    g_Leak=2., C=20., V1=-1.2, V2=18., V3=2., V4=30., phi=0.04):
    """
    The Morris-Lecar neuron model.

    ST refers to neuron state, members of ST are listed below:
    
    =============== ======== =========================================================
    **Member name** **Type** **Explanation**
    --------------- -------- ---------------------------------------------------------
    V               float    Membrane potential.
    
    W               float    Gating variable, refers to the fraction of opened K+ channels.
    
    input           float    External and synaptic input current.
    =============== ======== =========================================================

    Args:
        noise (float): The noise fluctuation.
        V_Ca (float): Equilibrium potentials of Ca+.(mV)
        g_Ca (float): Maximum conductance of corresponding Ca+.(mS/cm2)
        V_K (float): Equilibrium potentials of K+.(mV)
        g_K (float): Maximum conductance of corresponding K+.(mS/cm2)
        V_Leak (float): Equilibrium potentials of leak current.(mV)
        g_Leak (float): Maximum conductance of leak current.(mS/cm2)
        C (float): Membrane capacitance.(uF/cm2)
        V1 (float): Potential at which M_inf = 0.5.(mV)
        V2 (float): Reciprocal of slope of voltage dependence of M_inf.(mV)
        V3 (float): Potential at which W_inf = 0.5.(mV)
        V4 (float): Reciprocal of slope of voltage dependence of W_inf.(mV)
        phi (float): A temperature factor.(1/s)

    Returns:
        bp.Neutype: return description of Morris-Lecar model.
    """

    ST = bp.types.NeuState(
        {'V': -20, 'W': 0.02, 'input': 0.}
    )

    @bp.integrate
    def int_W(W, t, V):
        tau_W = 1 / (phi * np.cosh((V - V3) / (2 * V4)))
        W_inf = (1 / 2) * (1 + np.tanh((V - V3) / V4))
        dWdt = (W_inf - W) / tau_W
        return dWdt

    @bp.integrate
    def int_V(V, t, W, Isyn):
        M_inf = (1 / 2) * (1 + np.tanh((V - V1) / V2))
        I_Ca = g_Ca * M_inf * (V - V_Ca)
        I_K = g_K * W * (V - V_K)
        I_Leak = g_Leak * (V - V_Leak)
        dVdt = (- I_Ca - I_K - I_Leak + Isyn) / C
        return dVdt, noise / C

    def update(ST, _t_):
        W = int_W(ST['W'], _t_, ST['V'])
        V = int_V(ST['V'], _t_, ST['W'], ST['input'])
        ST['V'] = V
        ST['W'] = W
        ST['input'] = 0.

    def reset(ST):
        ST['input'] = 0.

    return bp.NeuType(name='MorrisLecar_neuron',
                      requires={"ST": ST},
                      steps=[update, reset],
                      vector_based=True)
