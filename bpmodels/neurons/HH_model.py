import brainpy as bp
import numpy as np

# constant values
E_NA = 50.
E_K = -77.
E_LEAK = -54.387
C = 1.0
G_NA = 120.
G_K = 36.
G_LEAK = 0.03
V_THRESHOLD = 20.

NOISE = 1.

def get_HH (noise=NOISE, V_th = V_THRESHOLD, C = C, E_Na = E_NA, E_K = E_K,
             E_leak = E_LEAK, g_Na = G_NA, g_K = G_K, g_leak = G_LEAK):
    '''
    A Hodgkin–Huxley neuron implemented in BrainPy.
    
    .. math::

        & C \\frac {dV} {dt} = -(\\bar{g}_{Na} m^3 h (V-E_{Na})  
        + \\bar{g}_K n^4 (V-E_K) + g_{leak} (V - E_{leak})) + I(t) 

        & \\frac {dx} {dt} = \\alpha_x (1-x)  - \\beta_x, \\quad x\\in \\ {\\rm{Na, K, leak}}

        & \\alpha_m(V) = \\frac {0.1(V+40)}{1-exp(\\frac{-(V + 40)} {10})}

        & \\beta_m(V) = 4.0 exp(\\frac{-(V + 65)} {18})

        & \\alpha_h(V) = 0.07 exp(\\frac{-(V+65)}{20})
        
        & \\beta_h(V) = \\frac 1 {1 + exp(\\frac{-(V + 35)} {10})}

        & \\alpha_n(V) = \\frac {0.01(V+55)}{1-exp(-(V+55)/10)}
        
        & \\beta_n(V) = 0.125 exp(\\frac{-(V + 65)} {80})


    ST refers to the neuron state, items in ST are listed below:
    
    =============== ==================  =========================================================
    **Member name** **Initial values**  **Explanation**
    --------------- ------------------  ---------------------------------------------------------
    V                        -65         Membrane potential.

    m                        0.05        gating variable of the sodium ion channel.

    n                        0.32        gating variable of the potassium ion channel.

    h                        0.60        gating variable of the sodium ion channel.

    input                     0          External and synaptic input current.

    spike                     0          Flag to mark whether the neuron is spiking. 
                                         Can be seen as bool.                                                      
    =============== ==================  =========================================================

    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        noise (float): the noise fluctuation.
        V_th (float): the spike threshold (mV).
        C (float): capacitance (ufarad).
        E_Na (float): reversal potential of sodium (mV).
        E_K (float): reversal potential of potassium (mV).
        E_leak (float): reversal potential of unspecific (mV).
        g_Na (float): conductance of sodium channel (msiemens).
        g_K (float): conductance of potassium channel (msiemens).
        g_leak (float): conductance of unspecific channels (msiemens).
        
    Returns:
        bp.NeuType

    References:
        .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description 
               of membrane current and its application to conduction and excitation 
               in nerve." The Journal of physiology 117.4 (1952): 500.

    '''
    
    # define variables and initial values
    ST = bp.types.NeuState(
        {'V': -65., 'm': 0.05, 'h': 0.60, 'n': 0.32, 'spike': 0., 'input': 0.},
        help='Hodgkin–Huxley neuron state.\n'
             '"V" denotes membrane potential.\n'
             '"n" denotes potassium channel activation probability.\n'
             '"m" denotes sodium channel activation probability.\n'
             '"h" denotes sodium channel inactivation probability.\n'
             '"spike" denotes spiking state.\n'
             '"input" denotes synaptic input.\n'
    )
    
    
    @bp.integrate
    def int_m(m, _t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m
    
    @bp.integrate
    def int_h(h, _t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h
    
    @bp.integrate
    def int_n(n, _t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n
    
    @bp.integrate
    def int_V(V, _t, m, h, n, I_ext):
        I_Na = (g_Na * np.power(m, 3.0) * h) * (V - E_Na)
        I_K = (g_K * np.power(n, 4.0))* (V - E_K)
        I_leak = g_leak * (V - E_leak)
        dVdt = (- I_Na - I_K - I_leak + I_ext)/C 
        return dVdt, noise / C
    
    # update the variables change over time (for each step)
    def update(ST, _t):
        m = np.clip(int_m(ST['m'], _t, ST['V']), 0., 1.)   # use np.clip to limit the int_m to between 0 and 1.
        h = np.clip(int_h(ST['h'], _t, ST['V']), 0., 1.)
        n = np.clip(int_n(ST['n'], _t, ST['V']), 0., 1.)
        V = int_V(ST['V'], _t, m, h, n, ST['input'])       # solve V from int_V equation.
        spike = np.logical_and(ST['V'] < V_th, V >= V_th)   # spike when reach threshold.
        ST['spike'] = spike
        ST['V'] = V
        ST['m'] = m
        ST['h'] = h
        ST['n'] = n

    def reset(ST):
        ST['input'] = 0.   
    
    return bp.NeuType(name='HH_neuron', 
                      ST=ST, 
                      steps=(update, reset), 
                      mode = 'vector')
