import npbrain as nb
import npbrain.numpy as np

# constant values
E_NA = 50.
E_K = -77.
E_LEAK = -54.387
C = 1.0
G_NA = 120.
G_K = 36.
G_LEAK = 0.03
V_THRESHOLD = 20.

NOISE = 0.

def HH (noise=NOISE, V_threshold = V_THRESHOLD, C = C, E_Na = E_NA, E_K = E_K,
             E_leak = E_LEAK, g_Na = G_NA, g_K = G_K, g_leak = G_LEAK):
    '''
    A Hodgkin–Huxley neuron implemented in NumpyBrain.
    
    Args:
        noise (float): the noise fluctuation. default = 0.
        V_threshold (float): the spike threshold. default = 20. (mV)
        C (float): capacitance. default = 1.0 (ufarad)
        E_Na (float): reversal potential of sodium. default = 50. (mV)
        E_K (float): reversal potential of potassium. default = -77. (mV)
        E_leak (float): reversal potential of unspecific. default = -54.387 (mV)
        g_Na (float): conductance of sodium channel. default = 120. (msiemens)
        g_K (float): conductance of potassium channel. default = 36. (msiemens)
        g_leak (float): conductance of unspecific channels. default = 0.03 (msiemens)
        
    Returns:
        HH_neuron (NeuType).
        
    '''
    
    # define variables and initial values
    ST = nb.types.NeuState(
        {'V': -65., 'm': 0.05, 'h': 0.60, 'n': 0.32, 'spike': 0., 'input': 0.},
        help='Hodgkin–Huxley neuron state.\n'
             '"V" denotes membrane potential.\n'
             '"n" denotes potassium channel activation probability.\n'
             '"m" denotes sodium channel activation probability.\n'
             '"h" denotes sodium channel inactivation probability.\n'
             '"spike" denotes spiking state.\n'
             '"input" denotes synaptic input.\n'
    )
    
    
    # call nb.integrate to solve the differential equations
    
    @nb.integrate
    def dmdt(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m
    
    @nb.integrate
    def dhdt(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h
    
    @nb.integrate
    def dndt(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n
    
    @nb.integrate(noise = noise / C)
    def dVdt(V, t, m, h, n, input_current):
        I_Na = (g_Na * np.power(m, 3.0) * h) * (V - E_Na)
        I_K = (g_K * np.power(n, 4.0))* (V - E_K)
        I_leak = g_leak * (V - E_K)
        dvdt = (- I_Na - I_K - I_leak + input_current)/C 
        return dvdt
    
    # update the variables change over time (for each step)
    def update(ST, _t_):
        m = np.clip(dmdt(ST['m'], _t_, ST['V']), 0., 1.) # use np.clip to limit the dmdt to between 0 and 1.
        h = np.clip(dhdt(ST['h'], _t_, ST['V']), 0., 1.)
        n = np.clip(dndt(ST['n'], _t_, ST['V']), 0., 1.)
        V = dVdt(ST['V'], _t_, m, h, n, ST['input'])  # solve V from dVdt equation.
        spike = np.logical_and(ST['V'] < V_threshold, V >= V_threshold) # spike when reach threshold.
        ST['spike'] = spike
        ST['V'] = V
        ST['m'] = m
        ST['h'] = h
        ST['n'] = n
        ST['input'] = 0.   
    
    return nb.NeuType(name='HH_neuron', requires={"ST": ST}, steps=update, vector_based=True)


def get_neuron(geometry, monitors=['spike', 'V', 'm', 'h', 'n'], pars_update = {}, **kwargs):
    '''
    Create HH neuron.

    Args:
        geometry (tuple): numbers of neurons to create, can be one or two dimensional.
        monitors (list): variables to record. Default = ['spike', 'V', 'm', 'h', 'n'].
        pars_update (dict): parameters to be updated. Default = {}.

    Returns:
        neuron (NeuGroup): type of neurons.
    '''
    if kwargs:
        pars_update = kwargs

    neuron = nb.NeuGroup(HH, geometry=geometry, 
                     monitors=monitors, 
                     pars_update=pars_update)
                        
    return neuron

def simulate(input_current, duration, geometry = (1,), monitors=['spike', 'V', 'm', 'h', 'n'], **kwargs):
    '''
    Apply input current to HH neurons.

    Args:
        input_current (NPArray): amplitude of the input current.
        duration (float): duration of the input current.
        geometry (tuple): numbers of neurons to create, can be one or two dimensional.
        monitors (list): variables to record. Default = ['spike', 'V', 'm', 'h', 'n'].

    Returns:
        (ts, V, m, h, n):
            ts (NPArray): time sequence of the simulation
            V (NPArray): membrane potential at each time step
            m (NPArray): m at each time step
            h (NPArray): h at each time step
            n (NPArray): n at each time step

    '''

    HH_neuron = get_neuron(geometry=geometry, monitors=monitors, pars_update = kwargs)

    net = nb.Network(HH_neuron)

    net.run(duration=duration, inputs=[HH_neuron, 'ST.input', input_current], report=True)

    ts = net.ts
    V = HH_neuron.mon.V[:, 0]
    m = HH_neuron.mon.m[:, 0]
    h = HH_neuron.mon.h[:, 0]
    n = HH_neuron.mon.n[:, 0]

    return (ts, V, m, h, n)
