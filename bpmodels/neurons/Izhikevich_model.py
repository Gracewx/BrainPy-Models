#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#This cell sets up the python libraries need for the code and plotting the output.
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.numpy as np


def get_Izhikevich(a=0.02, b=0.20, c=-65., d=8., t_refractory=0., noise=0., V_th=30., mode=None):
    """Izhikevich two-variable neuron model.
    Parameters
    ----------
    mode : optional, str
        The neuron spiking mode.
    a : float
        It determines the time scale of the recovery variable :math:`u`.
    b : float
        It describes the sensitivity of the recovery variable :math:`u` to
        the sub-threshold fluctuations of the membrane potential :math:`v`.
    c : float
        It describes the after-spike reset value of the membrane potential
        :math:`v` caused by the fast high-threshold :math:`K^{+}` conductance.
    d : float
        It describes after-spike reset of the recovery variable :math:`u` caused
        by slow high-threshold :math:`Na^{+}` and :math:`K^{+}` conductance.
    t_refractory : float
        Refractory period length. [ms]
    noise : float
        The noise fluctuation.
    V_th : float
        The membrane potential threshold.
    """
    state = bp.types.NeuState(
        {'V': -65., 'u': 1., 'spike': 0., 't_last_spike': -1e7, 'input': 0.},
        help='''
        Izhikevich two-variable neuron model state.
        V : membrane potential [mV].
        u : recovery variable [mV].
        spike : spike state. 
        t_last_spike : last spike time.
        inp : input, including external and synaptic inputs.
        '''
    )

    if mode in ['tonic', 'tonic spiking']:
        a, b, c, d = [0.02, 0.40, -65.0, 2.0]
    elif mode in ['phasic', 'phasic spiking']:
        a, b, c, d = [0.02, 0.25, -65.0, 6.0]
    elif mode in ['tonic bursting']:
        a, b, c, d = [0.02, 0.20, -50.0, 2.0]
    elif mode in ['phasic bursting']:
        a, b, c, d = [0.02, 0.25, -55.0, 0.05]
    elif mode in ['mixed mode']:
        a, b, c, d = [0.02, 0.20, -55.0, 4.0]
    elif mode in ['SFA', 'spike frequency adaptation']:
        a, b, c, d = [0.01, 0.20, -65.0, 8.0]
    elif mode in ['Class 1', 'class 1']:
        a, b, c, d = [0.02, -0.1, -55.0, 6.0]
    elif mode in ['Class 2', 'class 2']:
        a, b, c, d = [0.20, 0.26, -65.0, 0.0]
    elif mode in ['spike latency', ]:
        a, b, c, d = [0.02, 0.20, -65.0, 6.0]
    elif mode in ['subthreshold oscillation']:
        a, b, c, d = [0.05, 0.26, -60.0, 0.0]
    elif mode in ['resonator', ]:
        a, b, c, d = [0.10, 0.26, -60.0, -1.0]
    elif mode in ['integrator', ]:
        a, b, c, d = [0.02, -0.1, -55.0, 6.0]
    elif mode in ['rebound spike', ]:
        a, b, c, d = [0.03, 0.25, -60.0, 4.0]
    elif mode in ['rebound burst', ]:
        a, b, c, d = [0.03, 0.25, -52.0, 0.0]
    elif mode in ['threshold variability', ]:
        a, b, c, d = [0.03, 0.25, -60.0, 4.0]
    elif mode in ['bistability', ]:
        a, b, c, d = [1.00, 1.50, -60.0, 0.0]
    elif mode in ['DAP', 'depolarizing afterpotential']:
        a, b, c, d = [1.00, 0.20, -60.0, -21.0]
    elif mode in ['accommodation', ]:
        a, b, c, d = [0.02, 1.00, -55.0, 4.0]
    elif mode in ['inhibition-induced spiking', ]:
        a, b, c, d = [-0.02, -1.00, -60.0, 8.0]
    elif mode in ['inhibition-induced bursting', ]:
        a, b, c, d = [-0.026, -1.00, -45.0, 0]

    # Neurons
    elif mode in ['Regular Spiking', 'RS']:
        a, b, c, d = [0.02, 0.2, -65, 8]
    elif mode in ['Intrinsically Bursting', 'IB']:
        a, b, c, d = [0.02, 0.2, -55, 4]
    elif mode in ['Chattering', 'CH']:
        a, b, c, d = [0.02, 0.2, -50, 2]
    elif mode in ['Fast Spiking', 'FS']:
        a, b, c, d = [0.1, 0.2, -65, 2]
    elif mode in ['Thalamo-cortical', 'TC']:
        a, b, c, d = [0.02, 0.25, -65, 0.05]
    elif mode in ['Resonator', 'RZ']:
        a, b, c, d = [0.1, 0.26, -65, 2]
    elif mode in ['Low-threshold Spiking', 'LTS']:
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

        def update(ST, _t_):
            if (_t_ - ST['t_last_spike']) > t_refractory:
                V = int_V(ST['V'], _t_, ST['u'], ST['input'])
                u = int_u(ST['u'], _t_, ST['V'])
                if V >= V_th:
                    V = c
                    u += d
                    ST['t_last_spike'] = _t_
                    ST['spike'] = True
                ST['V'] = V
                ST['u'] = u
                ST['input'] = 0.
    else:

        def update(ST, _t_):
            V = int_V(ST['V'], _t_, ST['u'], ST['input'])
            u = int_u(ST['u'], _t_, ST['V'])
            if V >= V_th:
                V = c
                u += d
                ST['t_last_spike'] = _t_
                ST['spike'] = True
            ST['V'] = V
            ST['u'] = u
            ST['input'] = 0.

    return bp.NeuType(name='Izhikevich', requires={'ST': state}, steps=update, vector_based=False)

if __name__ == '__main__':
    bp.profile.set(backend='numba', merge_steps=True)

    Izhikevich = get_Izhikevich(mode='None')
    neu = bp.NeuGroup(Izhikevich, 10, monitors=['V', 'u'])
    neu.pars['noise'] = 0.

    current2 = bp.inputs.ramp_current(10, 10, 300, 0, 300)
    current1 = np.zeros(int(np.ceil(100 / 0.1)))
    current = np.append(current1, current2)
    neu.run(duration=400., inputs=['input', current], report=False)

    fig, gs = bp.visualize.get_figure(3, 1, 3, 8)

    fig.add_subplot(gs[0, 0])
    plt.plot(neu.mon.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, 400.1)
    plt.xlabel('Time (ms)')
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(neu.mon.ts, current, label='Input')
    plt.xlim(-0.1, 400.1)
    plt.ylim(0, 60)
    plt.ylabel('Input(mV)')
    plt.xlabel('Time (ms)')
    plt.legend()

    fig.add_subplot(gs[2, 0])
    plt.plot(neu.mon.ts, neu.mon.u[:, 0], label='u')
    plt.xlim(-0.1, 400.1)
    plt.ylabel('Recovery variable')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.show()

    if __name__ == '__main__':
        bp.profile.set(backend='numba', merge_steps=True)

        # tonic spiking
        Izhikevich = get_Izhikevich(mode='tonic spiking')
        neu.pars['noise'] = 0.
        current2 = bp.inputs.ramp_current(10, 10, 150, 0, 150)
        current1 = np.zeros(int(np.ceil(50 / 0.1)))
        current = np.append(current1, current2)
        neu.run(duration=200., inputs=['input', current], report=False)

        fig, ax1 = plt.subplots(figsize=(15, 5))
        plt.title('Tonic Spiking')
        ax1.plot(neu.mon.ts, neu.mon.V[:, 0], 'b', label='V')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Membrane potential (mV)', color='b')
        ax1.set_xlim(-0.1, 200.1)
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(neu.mon.ts, current, 'r', label='Input')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Input (mV)', color='r')
        ax2.set_ylim(0, 50)
        ax2.tick_params('y', colors='r')
        ax1.legend(loc=1)
        ax2.legend(loc=3)
        fig.tight_layout()
        plt.show()