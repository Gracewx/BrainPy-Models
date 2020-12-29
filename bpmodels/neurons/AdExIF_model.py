import brainpy as bp
import numpy as np
import sys

def get_AdExIF(V_rest=-65., V_reset=-68., V_th=-30., 
              V_T=-59.9, delta_T=3.48, a=1., b=1,
              R=1, C=10., tau=10., tau_w = 30.,
              t_refractory=0., noise=0., mode='scalar'):
    """Adaptive Exponential Integrate-and-Fire neuron model.
    
    .. math::
    
        \\tau_m\\frac{d V}{d t}= - (V-V_{rest}) + \\Delta_T e^{\\frac{V-V_T}{\\Delta_T}} - R w + RI(t)
    
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
        V_rest (float): Resting potential.
        V_reset (float): Reset potential after spike.
        V_th (float): Threshold potential of spike.
        V_T (float): Threshold potential of steady/non-steady.
        delta_T (float): Spike slope factor.
        R (float): Membrane resistance.
        C (float): Membrane capacitance.
        tau (float): Membrane time constant. Compute by R * C.
        tau_w (float): Time constant of the adaptation current.
        t_refractory (int): Refractory period length.
        noise (float): noise.   
        
    Returns:
        bp.Neutype: return description of ExpIF model.
    
    References:
        .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation 
               mechanisms determine the neuronal response to fluctuating 
               inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    """

    ST = bp.types.NeuState(
        {'V': 0, 'w':0, 'input': 0, 'spike': 0, 'refractory': 0, 't_last_spike': -1e7}
    )

    @bp.integrate
    def int_V(V, _t, w, I_ext):  # integrate u(t)
        return (- (V - V_rest) + delta_T * np.exp((V - V_T) / delta_T) - R * w + R * I_ext) / tau, noise / tau

    @bp.integrate
    def int_w(w, _t, V):
        return (a * (V - V_rest)-w) / tau_w, noise / tau_w

    def update(ST, _t):
        ST['spike'] = 0
        ST['refractory'] = True if _t - ST['t_last_spike'] <= t_refractory else False
        if not ST['refractory']:
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

    
    if mode == 'scalar':
        return bp.NeuType(name='AdExIF_neuron',
                          ST=ST,
                          steps=(update, reset),
                          mode=mode)
    elif mode == 'vector':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))