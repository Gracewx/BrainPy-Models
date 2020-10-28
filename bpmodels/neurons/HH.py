import brainpy as bp
import brainpy.numpy as np

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

def define_HH (noise=NOISE, V_threshold = V_THRESHOLD, C = C, E_Na = E_NA, E_K = E_K,
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
    
    
    # call bp.integrate to solve the differential equations
    
    @bp.integrate
    def dmdt(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m
    
    @bp.integrate
    def dhdt(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h
    
    @bp.integrate
    def dndt(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n
    
    @bp.integrate
    def dVdt(V, t, m, h, n, input_current):
        I_Na = (g_Na * np.power(m, 3.0) * h) * (V - E_Na)
        I_K = (g_K * np.power(n, 4.0))* (V - E_K)
        I_leak = g_leak * (V - E_K)
        dvdt = (- I_Na - I_K - I_leak + input_current)/C 
        return dvdt, noise / C
    
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
    
    return bp.NeuType(name='HH_neuron', requires={"ST": ST}, steps=update, vector_based=True)


def get_neuron(geometry, noise = NOISE, monitors=['spike', 'V', 'm', 'h', 'n'], 
             V_threshold = V_THRESHOLD, C = C, E_Na = E_NA, E_k = E_K,
             E_leak = E_LEAK, g_Na = G_NA, g_K = G_K, g_leak = G_LEAK):
    '''
    Create HH neuron.

    Args:
        geometry (tuple): numbers of neurons to create, can be one or two dimensional.
        monitors (list): variables to record. Default = ['spike', 'V', 'm', 'h', 'n'].

    Returns:
        neuron (NeuGroup): type of neurons.
    '''
    
    mode = bp.profile._backend

    if mode == 'numba':
        HH = define_HH(noise=noise)
    else:
        HH = define_HH(noise=noise, V_threshold = V_threshold, C = C, E_Na = E_Na, E_K = E_k,
             E_leak = E_leak, g_Na = g_Na, g_K = g_K, g_leak = g_leak)

    neuron = bp.NeuGroup(HH, geometry=geometry, monitors=monitors)

    if mode == 'numba':
        neuron.par['E_Na'] = E_Na
        neuron.par['E_K'] = E_k
        neuron.par['E_leak'] = E_leak
        neuron.par['g_Na'] = g_Na
        neuron.par['g_K'] = g_K
        neuron.par['g_leak'] = g_leak
        neuron.par['C'] = C
        neuron.par['V_threshold'] = V_threshold
                        
    return neuron

def simulate(input_current, duration, geometry = (1,), init_V = None, noise = NOISE,
             monitors=['spike', 'V', 'm', 'h', 'n'], V_threshold = V_THRESHOLD, 
             C = C, E_Na = E_NA, E_k = E_K, E_leak = E_LEAK, 
             g_Na = G_NA, g_K = G_K, g_leak = G_LEAK):
    '''
    Apply input current to HH neurons.

    Args:
        input_current (NPArray): amplitude of the input current.
        duration (float): duration of the input current.
        geometry (tuple): numbers of neurons to create, can be one or two dimensional.
        init_V (float): you can specify the intial value of membrane potential.
        monitors (list): variables to record. Default = ['spike', 'V', 'm', 'h', 'n'].

    Returns:
        (ts, V, m, h, n):
            ts (NPArray): time sequence of the simulation
            V (NPArray): membrane potential at each time step
            m (NPArray): m at each time step
            h (NPArray): h at each time step
            n (NPArray): n at each time step

    '''

    HH_neuron = get_neuron(geometry=geometry, noise = noise, monitors=monitors, 
                V_threshold = V_threshold, C = C, E_Na = E_Na, E_k = E_k,
                E_leak = E_leak, g_Na = g_Na, g_K = g_K, g_leak = g_leak)

    # set initial values
    if init_V == None:
        HH_neuron.ST['V'] = np.random.random(geometry) * 20 + -75
    else:
        HH_neuron.ST['V'] = init_V

    # run simulation
    HH_neuron.run(duration=duration, inputs=['ST.input', input_current], report=True)

    ts = HH_neuron.mon.ts
    V = HH_neuron.mon.V[:, 0]
    m = HH_neuron.mon.m[:, 0]
    h = HH_neuron.mon.h[:, 0]
    n = HH_neuron.mon.n[:, 0]

    return (ts, V, m, h, n)


if __name__ == "__main__":
    bp.profile.set(backend='numpy', dt=0.02, numerical_method='milstein', merge_steps=True)
    
    duration = 80.

    # Way 1
    (ts, V, m, h, n) = simulate(input_current = 5., duration = duration, geometry=1)

    '''
    # Way 2
    # HH_neuron = get_neuron(geometry=1, noise=0.)

    # Way 3
    HH = define_HH(noise = 0.)
    HH_neuron = bp.NeuGroup(HH, geometry=1, 
                    monitors=['spike', 'V', 'm', 'h', 'n'])


    HH_neuron.run(duration=duration, inputs=['ST.input', 5.], report=True)

    ts = HH_neuron.mon.ts
    V = HH_neuron.mon.V[:, 0]
    m = HH_neuron.mon.m[:, 0]
    h = HH_neuron.mon.h[:, 0]
    n = HH_neuron.mon.n[:, 0]
    '''

    # Visualization

    import matplotlib.pyplot as plt

    fig, gs = bp.visualize.get_figure(2, 1, 3, 8)

    # plot membrane potential
    fig.add_subplot(gs[0, 0])
    plt.plot(ts, V)
    plt.ylabel('Membrane potential (mV)')
    plt.xlim(-0.1, duration + 0.1)
    plt.title('Membrane potential')

    # plot gate variables
    fig.add_subplot(gs[1, 0])
    plt.plot(ts, m, label='m')
    plt.plot(ts, h, label='h')
    plt.plot(ts, n, label='n')
    plt.legend()
    plt.xlim(-0.1, duration + 0.1)
    plt.xlabel('Time (ms)')
    plt.ylabel('gate variables')
    plt.title('gate variables')

    plt.show()
