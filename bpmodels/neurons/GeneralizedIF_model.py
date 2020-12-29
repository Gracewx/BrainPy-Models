# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np
import sys

def get_GeneralizedIF(V_rest = -70., V_reset = -70., V_th_inf = -50., V_th_reset = -60.,
                      R = 20., C = 1., tau = 20., a = 0., b = 0.01, 
                      k1 = 0.2, k2 = 0.02, R1 = 0., R2 = 1., A1 = 0., A2 = 0.,
                      noise=0., mode='scalar'):
    """
    Generalized Integrate-and-Fire model (GeneralizedIF model).
    
    .. math::
    
        &\\frac{d I_j}{d t} = - k_j I_j
    
        &\\frac{d V}{d t} = ( - (V - V_{rest}) + R\\sum_{j}I_j + RI) / \\tau
    
        &\\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\\infty})
    
    When V meet Vth, Generalized IF neuron fire:
    
    .. math::
    
        &I_j \\leftarrow R_j I_j + A_j
    
        &V \\leftarrow V_{reset}
    
        &V_{th} \\leftarrow max(V_{th_{reset}}, V_{th})
    
    Note that I_j refers to arbitrary number of internal currents.
    
    ST refers to neuron state, members of ST are listed below:
    
    =============== ================= ==============================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ----------------------------------------------
    V               -70.              Membrane potential.
    
    input           0.                External and synaptic input current.
    
    spike           0.                Flag to mark whether the neuron is spiking. 
    
                                      Can be seen as bool.
    
    V_th            -50.              Spiking threshold potential.
                             
    I1              0.                Internal current 1.
    
    I2              0.                Internal current 2.
                             
    t_last_spike    -1e7              Last spike time stamp.
    =============== ================= ==============================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).
    
    Args:
        V_rest (float): Resting potential.
        V_reset (float): Reset potential after spike.
        V_th_inf (float): Target value of threshold potential V_th updating.
        V_th_reset (float): Free parameter, should be larger than V_reset.
        R (float): Membrane resistance.
        C (float): Membrane capacitance.
        tau (float): Membrane time constant. Compute by R * C.
        a (float): Coefficient describes the dependence of V_th on membrane potential.
        b (float): Coefficient describes V_th update.
        k1 (float): Constant pf I1.
        k2 (float): Constant of I2.
        R1 (float): Free parameter.
        R2 (float): Free parameter.
        A1 (float): Free parameter.
        A2 (float): Free parameter.
        noise (float): noise.   
        mode (str): Data structure of ST members.
        
    Returns:
        bp.Neutype: return description of Generalized IF model.
        
    References:
        .. [1] Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear 
               integrate-and-fire neural model produces diverse spiking 
               behaviors." Neural computation 21.3 (2009): 704-718.
    """

    ST = bp.types.NeuState(
        {'V': -70., 'input': 0., 'spike': 0., 'V_th': -50.,
         'I1': 0., 'I2': 0., 't_last_spike': -1e7}
    )
    
    @bp.integrate
    def int_I1(I1, _t):
        return - k1 * I1
        
    @bp.integrate
    def int_I2(I2, _t):
        return - k2 * I2
        
    @bp.integrate
    def int_V_th(V_th, _t, V):
        return a * (V- V_rest) - b * (V_th - V_th_inf)
    
    @bp.integrate
    def int_V(V, _t, I_ext, I1, I2):
        return ( - (V - V_rest) + R * I_ext + R * I1 + R * I2) / tau
        
    def update(ST, _t):
        ST['spike'] = 0
        I1 = int_I1(ST['I1'], _t)
        I2 = int_I2(ST['I2'], _t)
        V_th = int_V_th(ST['V_th'], _t, ST['V'])
        V = int_V(ST['V'], _t, ST['input'], ST['I1'], ST['I2'])
        if V > ST['V_th']:
            V = V_reset
            I1 = R1 * I1 + A1
            I2 = R2 * I2 + A2
            V_th = max(V_th, V_th_reset)
            ST['spike'] = 1
            ST['t_last_spike'] = _t
        ST['I1'] = I1
        ST['I2'] = I2
        ST['V_th'] = V_th
        ST['V'] = V
    
    def reset(ST):
        ST['input'] = 0.

    
    if mode == 'scalar':
        return bp.NeuType(name='GeneralizedIF_neuron',
                          ST=ST,
                          steps=(update, reset),
                          mode=mode)
    elif mode == 'vector':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
