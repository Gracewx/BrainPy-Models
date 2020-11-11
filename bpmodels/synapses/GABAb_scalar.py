# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.numpy as np
import bpmodels
from bpmodels.neurons import get_LIF

def get_GABAb_scalar(g_max=0.2, E=-95., k1=0.52, k2=0.0013, k3=0.098, k4=0.033, T=0.5, T_duration=0.3):
    '''
    GABAb synapse model.(scalar)
    
    .. math::

        &\\frac{d[R]}{dt} = k_3 [T](1-[R])- k_4 [R]

        &\\frac{d[G]}{dt} = k_1 [R]- k_2 [G]

        I_{GABA_{B}} &=\\overline{g}_{GABA_{B}} (\\frac{[G]^{4}} {[G]^{4}+100}) (V-E_{GABA_{B}})


    - [G] is the concentration of activated G protein.
    - [R] is the fraction of activated receptor.
    - [T] is the transmitter concentration.

    Args: 
        g_max (float): 
        E (float):
        k1 (float):
        k2 (float):
        k3 (float):
        k4 (float):
        T (float):
        T_duration (float):
    
    Returns:
        bp.Syntype: return description of GABAb model.
    '''

    requires = {
        'ST': bp.types.SynState({'R': 0., 'G': 0., 'g': 0., 't_last_spike': -1e7, }, 
                                help = 'GABAa synapse gating variable.'),
        'pre': bp.types.NeuState(['spike'], help = 
                                 'pre-synaptic neuron state must have "spike"'),
        'post': bp.types.NeuState(['input', 'V'], help = 
                                 'post-synaptic neuron state must include "input" and "V"')
    }

    @bp.integrate
    def int_R(R, t, TT):
        return k3 * TT * (1 - R) - k4 * R

    @bp.integrate
    def int_G(G, t, R):
        return k1 * R - k2 * G

    def update(ST, _t_, pre):
        if pre['spike'] > 0.:
            ST['t_last_spike'] = _t_
        TT = ((_t_ - ST['t_last_spike']) < T_duration) * T
        R = int_R(ST['R'], _t_, TT)
        G = int_G(ST['G'], _t_, R)
        ST['R'] = R
        ST['G'] = G
        ST['g'] = g_max * G ** 4 / (G ** 4 + 100)

    @bp.delayed
    def output(ST, _t_, post):
        I_syn = ST['g'] * (post['V'] - E)
        post['input'] -= I_syn

    return bp.SynType(name = 'GABAb_synapse', 
                      requires = requires, 
                      steps = (update, output), 
                      vector_based = False)
                      

if __name__ == '__main__':
    duration = 500.
    dt = 0.02
    bp.profile.set(backend = "numba", dt = dt, merge_steps = True, show_code = False)
    LIF_neuron = get_LIF()
    GABAb_syn = get_GABAb_scalar()
    
    # build and simulate gabab net
    pre = bp.NeuGroup(LIF_neuron, geometry = (10,), monitors = ['V', 'input', 'spike'])
    post = bp.NeuGroup(LIF_neuron, geometry = (10,), monitors = ['V', 'input', 'spike'])
    pre.runner.set_schedule(['input', 'update', 'monitor', 'reset'])
    pre.pars['V_rest'] = -65.
    pre.ST['V'] = -65.
    post.runner.set_schedule(['input', 'update', 'monitor', 'reset'])
    post.pars['V_rest'] = -65.
    post.ST['V'] = -65.
    
    gabab = bp.SynConn(model = GABAb_syn, pre_group = pre, post_group = post, 
                       conn = bp.connect.All2All(), monitors = ['g'], delay = 10.)
    gabab.runner.set_schedule(['input', 'update', 'output', 'monitor'])
    
    net = bp.Network(pre, gabab, post)
    
    current = bp.inputs.spike_current([5., 10., 15., 20., 25.], 
                                      bp.profile._dt, 1., duration = duration)
    net.run(duration = duration, inputs = [gabab, 'pre.spike', current, "="], report = True)

    
    
    #paint gabab
    ts = net.ts
    fig, gs = bp.visualize.get_figure(2, 2, 5, 6)
    
    fig.add_subplot(gs[0, 0])
    plt.plot(ts, gabab.mon.g[:, 0], label = 'g')
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(ts, post.mon.V[:, 0], label = 'post.V')
    plt.legend()
    
    fig.add_subplot(gs[0, 1])
    plt.plot(ts, post.mon.input[:, 0], label = 'post.input')
    plt.legend()
    
    plt.show()
