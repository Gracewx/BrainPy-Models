# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.numpy as np
import bpmodels
from bpmodels.neurons import get_LIF

def get_GABAb(g_max=0.2, E=-95., k1=0.52, k2=0.0013, k3=0.098, k4=0.033, T=0.5, T_duration=0.3):
    '''
    GABAb synapse model.
    '''

    requires = {
        'ST': bp.types.SynState({'R': 0., 'G': 0., 'sp_t': -1e7, 'g': 0.}, 
                                help = 'GABAa synapse gating variable.'),
        'pre': bp.types.NeuState(['isFire'], help = 
                                 'pre-synaptic neuron state must have "isFire"'),
        'post': bp.types.NeuState(['input', 'Vm'], help = 
                                 'post-synaptic neuron state must include "input" and "Vm"')
    }

    @bp.integrate
    def int_R(R, t, TT):
        return k3 * TT * (1 - R) - k4 * R

    @bp.integrate
    def int_G(G, t, R):
        return k1 * R - k2 * G

    def update(ST, _t_, pre):
        if pre['isFire'] > 0.:
            ST['sp_t'] = _t_
        TT = ((_t_ - ST['sp_t']) < T_duration) * T
        R = int_R(ST['R'], _t_, TT)
        G = int_G(ST['G'], _t_, R)
        ST['R'] = R
        ST['G'] = G
        ST['g'] = g_max * G ** 4 / (G ** 4 + 100)

    @bp.delayed
    def output(ST, _t_, post):
        I_syn = ST['g'] * (post['Vm'] - E)
        post['input'] -= I_syn

    return bp.SynType(name = 'GABAb', 
                      requires = requires, 
                      steps = (update, output), 
                      vector_based = False)
                      

if __name__ == '__main__':
    duration = 500.
    dt = 0.02
    bp.profile.set(backend = "numba", dt = dt, merge_steps = True, show_code = False)
    LIF_neuron = get_LIF()
    GABAb_syn = get_GABAb()
    
    # build and simulate gabab net
    pre = bp.NeuGroup(LIF_neuron, geometry = (10,), monitors = ['Vm', 'isFire', 'input'])
    post = bp.NeuGroup(LIF_neuron, geometry = (10,), monitors = ['Vm', 'isFire', 'input'])
    pre.runner.set_schedule(['input', 'update', 'monitor', 'reset'])
    pre.pars['Vr'] = -65.
    pre.ST['Vm'] = -65.
    post.runner.set_schedule(['input', 'update', 'monitor', 'reset'])
    post.pars['Vr'] = -65.
    post.ST['Vm'] = -65.
    
    gabab = bp.SynConn(model = GABAb_syn, pre_group = pre, post_group = post, 
                       conn = bp.connect.All2All(), monitors = ['g'], delay = 10.)
    gabab.runner.set_schedule(['input', 'update', 'output', 'monitor'])
    
    net = bp.Network(pre, gabab, post)
    
    current = bp.inputs.spike_current([5., 10., 15., 20., 25.], 
                                      bp.profile._dt, 1., duration = duration)
    net.run(duration = duration, inputs = [gabab, 'pre.isFire', current, "="], report = True)

    
    
    #paint gabab
    ts = net.ts
    fig, gs = bp.visualize.get_figure(2, 2, 5, 6)
    
    fig.add_subplot(gs[0, 0])
    plt.plot(ts, gabab.mon.g[:, 0], label = 'g')
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(ts, post.mon.Vm[:, 0], label = 'post.Vr')
    plt.legend()
    
    fig.add_subplot(gs[0, 1])
    plt.plot(ts, post.mon.input[:, 0], label = 'post.input')
    plt.legend()
    
    plt.show()
