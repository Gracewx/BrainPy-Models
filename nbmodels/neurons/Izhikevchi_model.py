# -*- coding: utf-8 -*-

#This cell sets up the python libraries need for the code and plotting the output.
import matplotlib.pyplot as plt
import npbrain as nb
import numpy as np


def Izhikevich(a=0.02, b=0.20, c=-65., d=8., ref=0., noise=0., Vth=30., Vr=-65., mode=None):
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
    ref : float
        Refractory period length. [ms]
    noise : float
        The noise fluctuation.
    Vth : float
        The membrane potential threshold.
    Vr : float
        The membrane reset potential.
    """
    state = nb.types.NeuState(
        {'V': -65., 'u': 1., 'sp': 0., 'sp_t': -1e7, 'inp': 0.},
        help='''
        Izhikevich two-variable neuron model state.
        V : membrane potential [mV].
        u : recovery variable [mV].
        sp : spike state. 
        sp_t : last spike time.
        inp : input, including external and synaptic inputs.
        '''
    )

    # Neuro-computational properties
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
        a, b, c, d = [-0.026, -1.00, -45.0, 0.0]

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

    @nb.integrate
    def int_u(u, t, V):
        return a * (b * V - u)

    @nb.integrate
    def int_V(V, t, u, Isyn):
        return 0.04 * V * V + 5 * V + 140 - u + Isyn

    if np.any(ref > 0.):

        def update(ST, _t_):
            V = int_V(ST['V'], _t_, ST['u'], ST['inp'])
            u = int_u(ST['u'], _t_, ST['V'])
            not_ref = (_t_ - ST['sp_t']) > ref
            for idx in np.where(np.logical_not(not_ref))[0]:
                V[idx] = ST['V'][idx]
                u[idx] = ST['u'][idx]
            sp = V >= Vth
            for idx in np.where(sp)[0]:
                V[idx] = c
                u[idx] += d
                ST['sp_t'] = _t_
            ST['sp'] = sp
            ST['V'] = V
            ST['u'] = u
            ST['inp'] = 0.
    else:

        def update(ST, _t_):
            V = int_V(ST['V'], _t_, ST['u'], ST['inp'])
            u = int_u(ST['u'], _t_, ST['V'])
            sp = V >= Vth
            spike_idx = np.where(sp)[0]
            for idx in spike_idx:
                V[idx] = c
                u[idx] += d
                ST['sp_t'] = _t_
            ST['V'] = V
            ST['u'] = u
            ST['sp'] = sp
            ST['inp'] = 0.

    return nb.NeuType(name='Izhikevich', requires={'ST': state}, steps=update, vector_based=True)

if __name__ == '__main__':
    nb.profile.set(backend='numba', )

    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0.,mode='None'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(10, 10, 300, 0, 300)
    current1 = np.zeros(int(np.ceil(100/0.1)))
    current = np.r_[current1, current2]
    net.run(duration=400., inputs=[neu, 'inp', current], report=False)

    fig, gs = nb.visualize.get_figure(3, 1, 3, 8)

    fig.add_subplot(gs[0, 0])
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(net.ts, current, label='Input')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.ylim(0, 60)
    plt.ylabel('Input(mV)')
    plt.xlabel('Time (ms)')
    plt.legend()

    fig.add_subplot(gs[2, 0])
    plt.plot(net.ts, neu.mon.u[:, 0], label='u')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.ylabel('Recovery variable')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    nb.profile.set(backend='numba', )
    fig, gs = nb.visualize.get_figure(3, 2, 3, 8)

    # tonic spiking
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0., mode='tonic spiking'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(10, 10, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50 / 0.1)))
    current = np.r_[current1, current2]
    net.run(duration=200., inputs=[neu, 'inp', current], report=False)

    fig.add_subplot(gs[0, 0])
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('tonic spiking')
    plt.legend()

    # phasic spiking
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0., mode='phasic spiking'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(1, 1, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50 / 0.1)))
    current = np.r_[current1, current2]
    net.run(duration=200., inputs=[neu, 'inp', current], report=False)

    fig.add_subplot(gs[0, 1])
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('phasic spiking')
    plt.legend()

    # tonic bursting
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0., mode='tonic bursting'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(15, 15, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50 / 0.1)))
    current = np.r_[current1, current2]
    net.run(duration=200., inputs=[neu, 'inp', current], report=False)

    fig.add_subplot(gs[1, 0])
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('tonic bursting')
    plt.legend()

    # phasic bursting
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0., mode='phasic bursting'),
                      monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(1, 1, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50 / 0.1)))
    current = np.r_[current1, current2]
    net.run(duration=200., inputs=[neu, 'inp', current], report=False)

    fig.add_subplot(gs[1, 1])
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('phasic bursting')
    plt.legend()

    # mixed mode
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0., mode='mixed mode'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(10, 10, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50 / 0.1)))
    current = np.r_[current1, current2]
    net.run(duration=200., inputs=[neu, 'inp', current], report=False)

    fig.add_subplot(gs[2, 0])
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('mixed mode')
    plt.legend()

    # spike frequency adaptation
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0., mode='spike frequency adaptation'),
                      monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(30, 30, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50 / 0.1)))
    current = np.r_[current1, current2]
    net.run(duration=200., inputs=[neu, 'inp', current], report=False)

    fig.add_subplot(gs[2, 1])
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('spike frequency adaptation')
    plt.legend()
    plt.show()

    plt.figure()
    fig, gs = nb.visualize.get_figure(1, 2, 3, 8)

    # Class 1
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0., mode='Class 1'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(0, 80, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50 / 0.1)))
    current = np.r_[current1, current2, current1]
    net.run(duration=250., inputs=[neu, 'inp', current], report=False)

    fig.add_subplot(gs[0, 0])
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('Class 1')
    plt.legend()

    # Class 2
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0., mode='Class 2'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(0, 10, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50 / 0.1)))
    current = np.r_[current1, current2, current1]
    net.run(duration=250., inputs=[neu, 'inp', current], report=False)

    fig.add_subplot(gs[0, 1])
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('Class 2')
    plt.legend()
    plt.show()

    #Regular Spiking (RS)
    plt.figure(figsize=(15,3))
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0.,mode='Regular Spiking'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(15, 15, 250, 0, 250)
    current1 = np.zeros(int(np.ceil(50/0.1)))
    current = np.r_[current1, current2]
    net.run(duration=300., inputs=[neu, 'inp', current], report=False)
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('Regular Spiking')
    plt.legend()
    plt.show()

    #Intrinsically Bursting (IB)
    plt.figure(figsize=(15,3))
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0.,mode='Intrinsically Bursting'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(15, 15, 250, 0, 250)
    current1 = np.zeros(int(np.ceil(50/0.1)))
    current = np.r_[current1, current2]
    net.run(duration=300., inputs=[neu, 'inp', current], report=False)
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('Intrinsically Bursting')
    plt.legend()
    plt.show()

    #Chattering (CH)
    plt.figure(figsize=(15,3))
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0.,mode='Chattering'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(10, 10, 350, 0, 350)
    current1 = np.zeros(int(np.ceil(50/0.1)))
    current = np.r_[current1, current2]
    net.run(duration=400., inputs=[neu, 'inp', current], report=False)
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('Chattering')
    plt.legend()
    plt.show()

    #Fast Spiking (FS)
    plt.figure(figsize=(15,3))
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0.,mode='Fast Spiking'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(10, 10, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50/0.1)))
    current = np.r_[current1, current2]
    net.run(duration=200., inputs=[neu, 'inp', current], report=False)
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('Fast Spiking')
    plt.legend()
    plt.show()

    #Thalamo-cortical (TC)
    plt.figure(figsize=(15,3))
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0.,mode='Thalamo-cortical'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(10, 10, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50/0.1)))
    current = np.r_[current1, current2]
    net.run(duration=200., inputs=[neu, 'inp', current], report=False)
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('Thalamo-cortical')
    plt.legend()
    plt.show()

    #Resonator (RZ)
    plt.figure(figsize=(15,3))
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0.,mode='Resonator'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(10, 10, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50/0.1)))
    current = np.r_[current1, current2]
    net.run(duration=200., inputs=[neu, 'inp', current], report=False)
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('Resonator')
    plt.legend()
    plt.show()

    #Low-threshold Spiking (LTS)
    plt.figure(figsize=(15,3))
    neu = nb.NeuGroup(Izhikevich, geometry=(1,), pars_update=dict(noise=0.,mode='Low-threshold Spiking'), monitors=['V', 'u'])
    net = nb.Network(neu)
    current2 = nb.inputs.ramp_current(10, 10, 150, 0, 150)
    current1 = np.zeros(int(np.ceil(50/0.1)))
    current = np.r_[current1, current2]
    net.run(duration=200., inputs=[neu, 'inp', current], report=False)
    plt.plot(net.ts, neu.mon.V[:, 0], label='V')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')
    plt.title('Low-threshold Spiking')
    plt.legend()
    plt.show()