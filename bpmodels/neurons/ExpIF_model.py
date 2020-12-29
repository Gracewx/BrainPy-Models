# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np
import sys

def get_ExpIF(V_rest=-65., V_reset=-68., V_th=-30., V_T=-59.9, delta_T=3.48,
              R=10., C=1., tau=10., t_refractory=1.7, noise=0., mode='scalar'):
    """Exponential Integrate-and-Fire neuron model.
    
    .. math::
    
        \\tau\\frac{d V}{d t}= - (V-V_{rest}) + \\Delta_T e^{\\frac{V-V_T}{\\Delta_T}} + RI(t)
    
    ST refers to neuron state, members of ST are listed below:
    
    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    V               0.                Membrane potential.
    
    input           0.                External and synaptic input current.
    
    spike           0.                Flag to mark whether the neuron is spiking. 
    
                                      Can be seen as bool.
                             
    refractory      0.                Flag to mark whether the neuron is in refractory period. 
     
                                      Can be seen as bool.
                             
    t_last_spike    -1e7              Last spike time stamp.
    =============== ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).
        
    Args:
        V_rest (float): Resting potential.
        V_reset (float): Reset potential after spike.
        V_th (float): Threshold potential of spike.
        V_T (float): Threshold potential of steady/non-steady.
        delta_T (float): Spike slope factor.
        R (float): Membrane resistance.
        C (float): Membrane capacitance.
        tau (float): Membrane time constant. Compute by R * C.
        t_refractory (int): Refractory period length.
        noise (float): noise.
        mode (str): Data structure of ST members.
        
    Returns:
        bp.Neutype: return description of ExpIF model.
    
    References:
        .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation 
               mechanisms determine the neuronal response to fluctuating 
               inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    """

    ST = bp.types.NeuState(
        {'V': 0, 'input': 0, 'spike': 0, 'refractory': 0, 't_last_spike': -1e7}
    )

    @bp.integrate
    def int_V(V, _t, I_ext):  # integrate u(t)
        return (- (V - V_rest) + delta_T * np.exp((V - V_T) / delta_T) + R * I_ext) / tau, noise / tau

    def update(ST, _t):
        # update variables
        ST['spike'] = 0
        ST['refractory'] = True if _t - ST['t_last_spike'] <= t_refractory else False
        if not ST['refractory']:
            V = int_V(ST['V'], _t, ST['input'])
            if V >= V_th:
                V = V_reset
                ST['spike'] = 1
                ST['t_last_spike'] = _t
            ST['V'] = V
            
    def reset(ST):
        ST['input'] = 0.

    if mode == 'scalar':
        return bp.NeuType(name='ExpIF_neuron',
                          ST=ST,
                          steps=(update, reset),
                          mode=mode)
    elif mode == 'vector':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
