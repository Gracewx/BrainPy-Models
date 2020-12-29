# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import brainpy as bp
import numpy as np
import sys

def get_Izhikevich(a=0.02, b=0.20, c=-65., d=8., t_refractory=0., noise=0., V_th=30., type=None, mode='scalar'):

    '''
    The Izhikevich neuron model.

    .. math ::

        \\frac{d V}{d t} &= 0.04 V^{2}+5 V+140-u+I

        \\frac{d u}{d t} &=a(b V-u)

    .. math ::
    
        \\text{if}  v \\geq 30  \\text{mV}, \\text{then}
        \\begin{cases} v \\leftarrow c \\\\ u \\leftarrow u+d \\end{cases}


    ST refers to neuron state, members of ST are listed below:
    
    =============== ======== ================== ===========================================
    **Member name** **Type** **Initial values** **Explanation**
    --------------- -------- ------------------ -------------------------------------------
    V               float            -65        Membrane potential.
    
    u               float            1          Recovery variable.
    
    input           float            0          External and synaptic input current.
    
    spike           float            0          Flag to mark whether the neuron is spiking. 
    
                                                Can be seen as bool.
                             
    t_last_spike    float            -1e7       Last spike time stamp.
    =============== ======== ================== ===========================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).    

    Args:
        type (str): The neuron spiking type.
        a (float): It determines the time scale of the recovery variable :math:`u`.
        b (float): It describes the sensitivity of the recovery variable :math:`u` to the sub-threshold fluctuations of the membrane potential :math:`v`.
        c (float): It describes the after-spike reset value of the membrane potential :math:`v` caused by the fast high-threshold :math:`K^{+}` conductance.
        d (float): It describes after-spike reset of the recovery variable :math:`u` caused by slow high-threshold :math:`Na^{+}` and :math:`K^{+}` conductance.
        t_refractory (float): Refractory period length. [ms]
        noise(float): The noise fluctuation.
        V_th (float): The membrane potential threshold.
        mode (str): Data structure of ST members.

    Returns:
        bp.Neutype: return description of Izhikevich model.


    References:
        .. [1] Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE
               Transactions on neural networks 14.6 (2003): 1569-1572.

        .. [2] Izhikevich, Eugene M. "Which model to use for cortical spiking neurons?." 
               IEEE transactions on neural networks 15.5 (2004): 1063-1070.


    Parameters of spiking types:
        
        =========================== ======= ======= ======= =======
        **Type**                     **a**   **b**   **c**   **d**
        --------------------------- ------- ------- ------- -------
        Regular Spiking              0.02    0.20    -65      8
        Intrinsically Bursting       0.02    0.20    -55      4
        Chattering                   0.02    0.20    -50      2
        Fast Spiking                 0.10    0.20    -65      2
        Thalamo-cortical             0.02    0.25    -65      0.05
        Resonator                    0.10    0.26    -65      2
        Low-threshold Spiking        0.02    0.25    -65      2
        tonic spiking                0.02    0.40    -65      2
        phasic spiking               0.02    0.25    -65      6
        tonic bursting               0.02    0.20    -50      2
        phasic bursting              0.02    0.25    -55      0.05
        mixed mode                   0.02    0.20    -55      4
        spike frequency adaptation   0.01    0.20    -65      8
        Class 1                      0.02    -0.1    -55      6
        Class 2                      0.20    0.26    -65      0
        spike latency                0.02    0.20    -65      6
        subthreshold oscillation     0.05    0.26    -60      0
        resonator                    0.10    0.26    -60      -1
        integrator                   0.02    -0.1    -55      6
        rebound spike                0.03    0.25    -60      4
        rebound burst                0.03    0.25    -52      0
        threshold variability        0.03    0.25    -60      4
        bistability                  1.00    1.50    -60      0
        depolarizing afterpotential  1.00    0.20    -60      -21
        accommodation                0.02    1.00    -55      4
        inhibition-induced spiking   -0.02   -1.00   -60      8
        inhibition-induced bursting  -0.026  -1.00   -45      0
        =========================== ======= ======= ======= =======
    '''

    ST = bp.types.NeuState(
           {'V': -65., 'u': 1., 'input': 0., 'spike': 0., 't_last_spike': -1e7}
    )

    if type in ['tonic', 'tonic spiking']:
        a, b, c, d = [0.02, 0.40, -65.0, 2.0]
    elif type in ['phasic', 'phasic spiking']:
        a, b, c, d = [0.02, 0.25, -65.0, 6.0]
    elif type in ['tonic bursting']:
        a, b, c, d = [0.02, 0.20, -50.0, 2.0]
    elif type in ['phasic bursting']:
        a, b, c, d = [0.02, 0.25, -55.0, 0.05]
    elif type in ['mixed mode']:
        a, b, c, d = [0.02, 0.20, -55.0, 4.0]
    elif type in ['SFA', 'spike frequency adaptation']:
        a, b, c, d = [0.01, 0.20, -65.0, 8.0]
    elif type in ['Class 1', 'class 1']:
        a, b, c, d = [0.02, -0.1, -55.0, 6.0]
    elif type in ['Class 2', 'class 2']:
        a, b, c, d = [0.20, 0.26, -65.0, 0.0]
    elif type in ['spike latency', ]:
        a, b, c, d = [0.02, 0.20, -65.0, 6.0]
    elif type in ['subthreshold oscillation']:
        a, b, c, d = [0.05, 0.26, -60.0, 0.0]
    elif type in ['resonator', ]:
        a, b, c, d = [0.10, 0.26, -60.0, -1.0]
    elif type in ['integrator', ]:
        a, b, c, d = [0.02, -0.1, -55.0, 6.0]
    elif type in ['rebound spike', ]:
        a, b, c, d = [0.03, 0.25, -60.0, 4.0]
    elif type in ['rebound burst', ]:
        a, b, c, d = [0.03, 0.25, -52.0, 0.0]
    elif type in ['threshold variability', ]:
        a, b, c, d = [0.03, 0.25, -60.0, 4.0]
    elif type in ['bistability', ]:
        a, b, c, d = [1.00, 1.50, -60.0, 0.0]
    elif type in ['DAP', 'depolarizing afterpotential']:
        a, b, c, d = [1.00, 0.20, -60.0, -21.0]
    elif type in ['accommodation', ]:
        a, b, c, d = [0.02, 1.00, -55.0, 4.0]
    elif type in ['inhibition-induced spiking', ]:
        a, b, c, d = [-0.02, -1.00, -60.0, 8.0]
    elif type in ['inhibition-induced bursting', ]:
        a, b, c, d = [-0.026, -1.00, -45.0, 0]

    # Neurons
    elif type in ['Regular Spiking', 'RS']:
        a, b, c, d = [0.02, 0.2, -65, 8]
    elif type in ['Intrinsically Bursting', 'IB']:
        a, b, c, d = [0.02, 0.2, -55, 4]
    elif type in ['Chattering', 'CH']:
        a, b, c, d = [0.02, 0.2, -50, 2]
    elif type in ['Fast Spiking', 'FS']:
        a, b, c, d = [0.1, 0.2, -65, 2]
    elif type in ['Thalamo-cortical', 'TC']:
        a, b, c, d = [0.02, 0.25, -65, 0.05]
    elif type in ['Resonator', 'RZ']:
        a, b, c, d = [0.1, 0.26, -65, 2]
    elif type in ['Low-threshold Spiking', 'LTS']:
        a, b, c, d = [0.02, 0.25, -65, 2]

    @bp.integrate
    def int_u(u, t, V):
        return a * (b * V - u)

    @bp.integrate
    def int_V(V, t, u, Isyn):
        dfdt = 0.04 * V * V + 5 * V + 140 - u + Isyn
        dgdt = noise
        return dfdt, dgdt

    if np.any(t_refractory > 0.):

        def update(ST, _t):
            if (_t - ST['t_last_spike']) > t_refractory:
                V = int_V(ST['V'], _t, ST['u'], ST['input'])
                u = int_u(ST['u'], _t, ST['V'])
                if V >= V_th:
                    V = c
                    u += d
                    ST['t_last_spike'] = _t
                    ST['spike'] = True
                ST['V'] = V
                ST['u'] = u
    else:

        def update(ST, _t):
            V = int_V(ST['V'], _t, ST['u'], ST['input'])
            u = int_u(ST['u'], _t, ST['V'])
            if V >= V_th:
                V = c
                u += d
                ST['t_last_spike'] = _t
                ST['spike'] = True
            ST['V'] = V
            ST['u'] = u

    def reset(ST):
        ST['input'] = 0.



    if mode == 'scalar':
        return bp.NeuType(name='Izhikevich_neuron',
                          ST=ST,
                          steps=(update, reset),
                          mode=mode)
    elif mode == 'vector':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
