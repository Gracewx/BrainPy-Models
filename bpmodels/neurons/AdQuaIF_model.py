import brainpy as bp
import sys

def get_AdQuaIF(a = 1, b = .1, a_0 = .07,  
                    V_c = -50, V_rest=-65., V_reset=-68., V_th=-30.,
                    R=1., C=10.,
                    tau=10., tau_w = 10., 
                    t_refractory=0., noise=0., mode='scalar'):
    """Adaptive Quadratic Integrate-and-Fire neuron model.
        
    .. math::

        \\tau_m \\frac{d V}{d t}=a_0(V-V_{rest})(V-V_c) - R w + RI(t)
        
        \\tau_w \\frac{d w}{d t}=a(V-V_{rest}) - w + b \\tau_w \\sum \\delta (t-t^f)
    
    ST refers to neuron state, members of ST are listed below:
    
    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    V               0.                Membrane potential.

    w               0.                Adaptation current.
    
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
        a (float):
        b (float):
        a_0 (float): Coefficient describes membrane potential update. Larger than 0.
        V_c (float): Critical voltage for spike initiation. Must be larger than V_rest.
        V_rest (float): Resting potential.
        V_reset (float): Reset potential after spike.
        V_th (float): Threshold potential of spike.
        R (float): Membrane resistance.
        C (float): Membrane capacitance.
        tau (float): Membrane time constant. Compute by R * C.
        tau_w (float): Time constant of the adaptation current.
        t_refractory (int): Refractory period length.(ms)
        noise (float): noise.   
        
    Returns:
        bp.Neutype: return description of QuaIF model.
        
    References:
        .. [1] Gerstner, Wulfram, et al. Neuronal dynamics: From single 
               neurons to networks and models of cognition. Cambridge 
               University Press, 2014.
    """

    if mode == 'vector':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    elif mode != 'scalar':
        raise ValueError("BrainPy does not support mode '%s'." % (mode))


    ST = bp.types.NeuState(
        {'V': 0, 'w':0, 'input': 0, 'spike': 0, 'refractory': 0, 't_last_spike': -1e7}
    )

    @bp.integrate
    def int_V(V, _t, w, I_ext):  
        return (a_0* (V - V_rest)*(V-V_c) - R * w + R * I_ext) / tau, noise / tau

    @bp.integrate
    def int_w(w, _t, V):
        return (a* (V - V_rest)-w) / tau_w, noise / tau_w

    def update(ST, _t):
        ST['spike'] = 0
        if _t - ST['t_last_spike'] <= t_refractory:
            ST['refractory'] = 1.
        else:
            ST['refractory'] = 0.
            w = int_w(ST['w'], _t, ST['V'])
            V = int_V(ST['V'], _t, w, ST['input'])
            if V >= V_th:
                V = V_reset
                w += b
                ST['spike'] = 1
                ST['t_last_spike'] = _t
            ST['V'] = V
            ST['w'] = w

    def reset(ST):
        ST['input'] = 0.

    return bp.NeuType(name='AdQuaIF_neuron',
                      ST=ST,
                      steps=(update, reset),
                      mode=mode)    