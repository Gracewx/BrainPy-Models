# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.numpy as np
import bpmodels
from bpmodels.neurons import get_LIF

def get_GABAa(g_max = 0.4, E = -80., tau_decay = 6.):
    '''
    GABAa synapse model.
    '''

    requires = {
        'ST': bp.types.SynState(['s'], 
                                help = 'GABAa synapse gating variable.'),
        'pre': bp.types.NeuState(['isFire'], help = 
                                 'pre-synaptic neuron state must have "isFire"'),
        'post': bp.types.NeuState(['input', 'Vm'], help = 
                                 'post-synaptic neuron state must include "input" and "Vr"')
    }

    @bp.integrate
    def int_s(s, _t_):
        return -s / tau_decay

    def update(ST, _t_, pre):
        s = int_s(ST['s'], _t_)
        s += pre['isFire']
        ST['s'] = s

    @bp.delayed
    def output(ST, _t_, post):
        I_syn = - g_max * ST['s'] * (post['Vm'] - E)
        post['input'] += I_syn

    return bp.SynType(name = 'GABAa', 
                      requires = requires, 
                      steps = (update, output), 
                      vector_based = False)
                      
if __name__ == '__main__':
    duration = 500.
    dt = 0.02
    bp.profile.set(backend = "numba", dt = dt, merge_steps = True, show_code = False)
    LIF_neuron = get_LIF()
    GABAa_syn = get_GABAa()

    #build and simulate gabaa net
    pre = bp.NeuGroup(LIF_neuron, geometry = (10,), monitors = ['Vm', 'isFire', 'input'])
    pre.runner.set_schedule(['input', 'update', 'monitor', 'reset'])
    pre.pars['Vr'] = -65.
    pre.ST['Vm'] = -65.
    post = bp.NeuGroup(LIF_neuron, geometry = (10,), monitors = ['Vm', 'isFire', 'input'])
    post.runner.set_schedule(['input', 'update', 'monitor', 'reset'])
    post.pars['Vr'] = -65.
    post.ST['Vm'] = -65.

    gabaa = bp.SynConn(model = GABAa_syn, pre_group = pre, post_group = post, 
                       conn = bp.connect.All2All(), monitors = ['s'], delay = 10.)
    gabaa.runner.set_schedule(['input', 'update', 'output', 'monitor'])
    
    net = bp.Network(pre, gabaa, post)
    
    current = bp.inputs.spike_current([10, 110, 210, 300, 305, 310, 315, 320], 
                                      bp.profile._dt, 1., duration = duration)
    net.run(duration = duration, inputs = [gabaa, 'pre.isFire', current, "="], report = True)

    # paint gabaa
    ts = net.ts
    fig, gs = bp.visualize.get_figure(2, 2, 5, 6)
    
    fig.add_subplot(gs[0, 0])
    plt.plot(ts, gabaa.mon.s[:, 0], label = 's')
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(ts, post.mon.Vm[:, 0], label = 'post.Vr')
    plt.legend()
    
    fig.add_subplot(gs[0, 1])
    plt.plot(ts, post.mon.input[:, 0], label = 'post.input')
    plt.legend()
    
    plt.show()
    
