#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.numpy as np

def get_MorrisLecar(noise=0., V_Ca=120., g_Ca=4.4, V_K=-84., g_K=8., V_Leak=-60.,
                    g_Leak=2., C=20., V1=-1.2, V2=18., V3=2., V4=30., phi=0.04):

    """The MorrisLecar neuron model.
    Parameters
    ----------
    noise : float
        The noise fluctuation.
    V_Ca : float
    g_Ca : float
    V_K : float
    g_K : float
    V_Leak : float
    g_Leak : float
    C : float
    V1 : float
    V2 : float
    V3 : float
    V4 : float
    phi : float
    Returns
    -------
    return_dict : dict
        The necessary variables.
    """

    ST = bp.types.NeuState(
        {'V': -20, 'W': 0.02, 'input': 0.},
        help='MorrisLecar model neuron state.\n'
             '"V" denotes membrane potential.\n'
             '"W" denotes fraction of open K+ channels.\n'
             '"input" denotes synaptic input.\n'
    )

    @bp.integrate
    def int_W(W, t, V):
        tau_W = 1 / (phi * np.cosh((V - V3) / (2 * V4)))
        W_inf = (1 / 2) * (1 + np.tanh((V - V3) / V4))
        dWdt = (W_inf - W) / tau_W
        return dWdt

    @bp.integrate
    def int_V(V, t, W, Isyn):
        M_inf = (1 / 2) * (1 + np.tanh((V - V1) / V2))
        I_Ca = g_Ca * M_inf * (V - V_Ca)
        I_K = g_K * W * (V - V_K)
        I_Leak = g_Leak * (V - V_Leak)
        dVdt = (- I_Ca - I_K - I_Leak + Isyn) / C
        return dVdt, noise / C

    def update(ST, _t_):
        W = int_W(ST['W'], _t_, ST['V'])
        V = int_V(ST['V'], _t_, ST['W'], ST['input'])
        ST['V'] = V
        ST['W'] = W
        ST['input'] = 0.

    return bp.NeuType(name='MorrisLecar_neuron', requires={"ST": ST}, steps=update, vector_based=True)


if __name__ == '__main__':
    bp.profile.set(backend='numba', dt=0.02, merge_steps=True)
    ML = get_MorrisLecar(noise=0.)

    #The current is constant
    neu = bp.NeuGroup(ML, geometry=(100,), monitors=['V', 'W'])
    current = bp.inputs.ramp_current(90, 90, 1000, 0, 1000)
    neu.run(duration=1000., inputs=['ST.input', current], report=False)

    fig, gs = bp.visualize.get_figure(2, 2, 3, 6)
    fig.add_subplot(gs[0, 0])
    plt.plot(neu.mon.V[:, 0], neu.mon.W[:, 0], label='V')
    plt.xlabel('Membrane potential (mV)')
    plt.ylabel('Recovery Variable')
    plt.title('W - V')
    plt.legend()

    fig.add_subplot(gs[0, 1])
    plt.plot(neu.mon.ts, neu.mon.V[:, 0], label='V')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential')
    plt.title('V - t')
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(neu.mon.ts, neu.mon.W[:, 0], label='W')
    plt.xlabel('Time (ms)')
    plt.ylabel('Recovery Variable')
    plt.title('W - t')
    plt.legend()

    fig.add_subplot(gs[1, 1])
    plt.plot(neu.mon.ts, current, label='input')
    plt.xlabel('Time (ms)')
    plt.ylabel('Input')
    plt.title('Input - t')
    plt.legend()

    plt.show()