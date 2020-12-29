# -*- coding: utf-8 -*-
import brainpy as bp
import numpy as np
import sys


def get_MorrisLecar(noise=0., V_Ca=130., g_Ca=4.4, V_K=-84., g_K=8., V_Leak=-60.,
                    g_Leak=2., C=20., V1=-1.2, V2=18., V3=2., V4=30., phi=0.04, mode='vector'):
    """
    The Morris-Lecar neuron model. (Also known as :math:`I_{Ca}+I_K`-model.)

    .. math::

        &C\\frac{dV}{dt} = -  g_{Ca} M_{\\infty} (V - V_{Ca})- g_{K} W(V - V_{K}) - g_{Leak} (V - V_{Leak}) + I_{ext}

        &\\frac{dW}{dt} = \\frac{W_{\\infty}(V) - W}{ \\tau_W(V)} 

    ST refers to neuron state, members of ST are listed below:
    
    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    V               -20.              Membrane potential.
    
    W               0.02              Gating variable, refers to the fraction of 
                                      opened K+ channels.
    
    input           0.                External and synaptic input current.
    =============== ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

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
        mode (str): Data structure of ST members.

    Returns:
        bp.Neutype: return description of Morris-Lecar model.

    References:
        .. [1] Meier, Stephen R., Jarrett L. Lancaster, and Joseph M. Starobin.
               "Bursting regimes in a reaction-diffusion system with action 
               potential-dependent equilibrium." PloS one 10.3 (2015): 
               e0122401.
    """

    ST = bp.types.NeuState(
        {'V': -20, 'W': 0.02, 'input': 0.}
    )

    @bp.integrate
    def int_W(W, _t, V):
        tau_W = 1 / (phi * np.cosh((V - V3) / (2 * V4)))
        W_inf = (1 / 2) * (1 + np.tanh((V - V3) / V4))
        dWdt = (W_inf - W) / tau_W
        return dWdt

    @bp.integrate
    def int_V(V, _t, W, Isyn):
        M_inf = (1 / 2) * (1 + np.tanh((V - V1) / V2))
        I_Ca = g_Ca * M_inf * (V - V_Ca)
        I_K = g_K * W * (V - V_K)
        I_Leak = g_Leak * (V - V_Leak)
        dVdt = (- I_Ca - I_K - I_Leak + Isyn) / C
        return dVdt, noise / C

    def update(ST, _t):
        W = int_W(ST['W'], _t, ST['V'])
        V = int_V(ST['V'], _t, ST['W'], ST['input'])
        ST['V'] = V
        ST['W'] = W
        ST['input'] = 0.

    def reset(ST):
        ST['input'] = 0.

                             
    if mode == 'scalar':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    elif mode == 'vector':
        return bp.NeuType(name='MorrisLecar_neuron',
                          ST=ST,
                          steps=[update, reset],
                          mode=mode)
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
