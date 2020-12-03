# -*- coding: utf-8 -*-

import brainpy as bp

def get_GeneralizedIF(V_rest = -70., V_reset = -70., V_th_inf = -50., V_th_reset = -60.,
                      R = 20., C = 1., tau = 20., a = 0., b = 0.01, 
                      k1 = 0.2, k2 = 0.02, R1 = 0., R2 = 1., A1 = 0., A2 = 0.,
                      noise=0., 
                      ):
    """
    Generalized Integrate-and-Fire model (GeneralizedIF model).
    
    .. math::
    
        &\\frac{d I_j}{d t} = - k_j I_j
    
        &\\frac{d V}{d t} = ( - (V - V_{rest}) + R\\sum_{j}I_j + RI)
    
        &\\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\\infty})
    
    When V meet Vth, Generalized IF neuron fire:
    
    .. math::
    
        &I_j \\leftarrow R_j I_j + A_j
    
        &V \\leftarrow V_{reset}
    
        &V_{th} \\leftarrow max(V_{th_{reset}}, V_{th})
    
    Note that I_j refers to arbitrary number of internal currents.
    
    ST refers to neuron state, members of ST are listed below:
    
    =============== ======== =========================================================
    **Member name** **Type** **Explanation**
    --------------- -------- ---------------------------------------------------------
    V               float    Membrane potential.
    
    input           float    External and synaptic input current.
    
    spike           float    Flag to mark whether the neuron is spiking. 
    
                             Can be seen as bool.
    
    V_th            float    Spiking threshold potential.
                             
    I1              float    Internal current 1.
    
    I2              float    Internal current 2.
                             
    t_last_spike    float    Last spike time stamp.
    =============== ======== =========================================================
    
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
    def int_I1(I1, _t_):
        return - k1 * I1
        
    @bp.integrate
    def int_I2(I2, _t_):
        return - k2 * I2
        
    @bp.integrate
    def int_V_th(V_th, _t_, V):
        return a * (V- V_rest) - b * (V_th - V_th_inf)
    
    @bp.integrate
    def int_V(V, _t_, I_ext, I1, I2):
        return ( - (V - V_rest) + R * I_ext + R * I1 + R * I2) / tau
        
    def update(ST, _t_):
        #is there refractory??
        ST['spike'] = 0
        I1 = int_I1(ST['I1'], _t_)
        I2 = int_I2(ST['I2'], _t_)
        V_th = int_V_th(ST['V_th'], _t_, ST['V'])
        V = int_V(ST['V'], _t_, ST['input'], ST['I1'], ST['I2'])
        if V > ST['V_th']:
            V = V_reset
            I1 = R1 * I1 + A1
            I2 = R2 * I2 + A2
            V_th = max(V_th, V_th_reset)
            ST['spike'] = 1
            ST['t_last_spike'] = _t_
        ST['I1'] = I1
        ST['I2'] = I2
        ST['V_th'] = V_th
        ST['V'] = V
    
    def reset(ST):
        ST['input'] = 0.

    return bp.NeuType(name='GeneralizedIF_neuron',
                      requires=dict(ST=ST),
                      steps=(update, reset),
                      vector_based=False)