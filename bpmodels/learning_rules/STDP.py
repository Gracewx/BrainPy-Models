# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import brainpy as bp
from brainpy import numpy as np
import bpmodels
from bpmodels.neurons import get_LIF


def get_STDP1(g_max=0.10, E=0., tau_decay=10., tau_s = 10., tau_t = 10., 
                   w_min = 0., w_max = 20., delta_A_s = 0.5, delta_A_t = 0.5):
    """
    Spike-time dependent plasticity (in differential form).
    
    .. math::

        \\frac{d A_{source}}{d t}&=-\\frac{A_{source}}{\\tau_{source}}
        
        \\frac{d A_{target}}{d t}&=-\\frac{A_{target}}{\\tau_{target}}
    
    After a pre-synaptic spike:

    .. math::      
      
        g_{post}&= g_{post}+w
        
        A_{source}&= A_{source} + \\delta A_{source}
        
        w&= min([w+A_{target}]^+, w_{max})
        
    After a post-synaptic spike:
    
    .. math::
        
        A_{target}&= A_{target} + \\delta A_{target}
        
        w&= min([w+A_{source}]^+, w_{max})
    
    Args:
        g_max (float): Maximum conductance.
        E (float): Reversal potential.
        tau_decay (float): Time constant of decay.
        tau_s (float): Time constant of source neuron (i.e. pre-synaptic neuron)
        tau_t (float): Time constant of target neuron (i.e. post-synaptic neuron)
        w_min (float): Minimal possible synapse weight.
        w_max (float): Maximal possible synapse weight.
        delta_A_s (float): Change on source neuron traces elicited by a source neuron spike.
        delta_A_t (float): Change on target neuron traces elicited by a target neuron spike.
        
    Returns:
        bp.Syntype: return description of STDP.
        
    References:
        .. [1] Stimberg, Marcel, et al. "Equation-oriented specification of neural models for
               simulations." Frontiers in neuroinformatics 8 (2014): 6.
    """

    requires = dict(
        ST=bp.types.SynState(['A_s', 'A_t', 'g', 'w'], help='AMPA synapse state.'),
        pre=bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "spike" item.'),
        post=bp.types.NeuState(['V', 'input', 'spike'], help='Post-synaptic neuron state must have "V", "input" and "spike" item.'),
        pre2syn=bp.types.ListConn(
            help='Pre-synaptic neuron index -> synapse index'),
        post2syn=bp.types.ListConn(
            help='Post-synaptic neuron index -> synapse index'),
            
    )

    @bp.integrate
    def int_A_s(A_s, _t_):
        return -A_s / tau_s

    @bp.integrate
    def int_A_t(A_t, _t_):
        return -A_t / tau_t

    @bp.integrate
    def int_g(g, _t_):
        return -g / tau_decay

    def my_relu(w):
        #return w if w > 0 else 0  # scalar
        for i in range(len(w)):    # vector
            w[i] = w[i] if w[i] > 0 else 0
        return w

    def update(ST, _t_, pre, post, pre2syn, post2syn):
        A_s = int_A_s(ST['A_s'], _t_)
        A_t = int_A_t(ST['A_t'], _t_)
        g = int_g(ST['g'], _t_)
        w = ST['w']
        
        for i in np.where(pre['spike'] > 0.)[0]:
            syn_ids = pre2syn[i]
            g[syn_ids] += ST['w'][syn_ids]
            A_s[syn_ids] = A_s[syn_ids] + delta_A_s
            w[syn_ids] = np.clip(my_relu(ST['w'][syn_ids] - A_t[syn_ids]), w_min, w_max)
        for i in np.where(post['spike'] > 0.)[0]:
            syn_ids = post2syn[i]
            A_t[syn_ids] = A_t[syn_ids] + delta_A_t
            w[syn_ids] = np.clip(my_relu(ST['w'][syn_ids] + A_s[syn_ids]), w_min, w_max)
        ST['A_s'] = A_s
        ST['A_t'] = A_t
        ST['g'] = g
        ST['w'] = w

    @bp.delayed
    def output(ST, post, post2syn):
        #I_syn = - g_max * ST['g'] * (post['V'] - E)

        #post['input'] += I_syn
            
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(- g_max * ST['g'][syn_ids] * (post['V'] - E))
        post['input'] += post_cond

    return bp.SynType(name='STDP_AMPA_synapse',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)
                      
                      
def get_STDP2(g_max=0.10, E=0., tau_decay=10., tau_s = 10., tau_t = 10., 
                   w_min = 0., w_max = 20., delta_A_s = 0.5, delta_A_t = 0.5):
    """
    Spike-time dependent plasticity (in inetgrated form).
    
    After a pre-synaptic spike:

    .. math::      
      
        g_{post}&= g_{post}+w
        
        A_{source}&= A_{source}*e^{\\frac{last_update-t}{\\tau_{source}}} + \\delta A_{source}
        
        A_{target}&= A_{target}*e^{\\frac{last_update-t}{\\tau_{target}}}
        
        w&= min([w+A_{target}]^+, w_{max})
        
    After a post-synaptic spike:
    
    .. math::
        
        A_{source}&= A_{source}*e^{\\frac{last_update-t}{\\tau_{source}}}
        
        A_{target}&= A_{target}*e^{\\frac{last_update-t}{\\tau_{target}}} + \\delta A_{target}
        
        w&= min([w+A_{source}]^+, w_{max})
    
    Args:
        g_max (float): Maximum conductance.
        E (float): Reversal potential.
        tau_decay (float): Time constant of decay.
        tau_s (float): Time constant of source neuron (i.e. pre-synaptic neuron)
        tau_t (float): Time constant of target neuron (i.e. post-synaptic neuron)
        w_min (float): Minimal possible synapse weight.
        w_max (float): Maximal possible synapse weight.
        delta_A_s (float): Change on source neuron traces elicited by a source neuron spike.
        delta_A_t (float): Change on target neuron traces elicited by a target neuron spike.
        
    Returns:
        bp.Syntype: return description of STDP.
        
    References:
        .. [1] Stimberg, Marcel, et al. "Equation-oriented specification of neural models for
               simulations." Frontiers in neuroinformatics 8 (2014): 6.
    """

    requires = dict(
        ST=bp.types.SynState(['A_s', 'A_t', 'g', 'w', 'last_spike'], help='AMPA synapse state.'),
        pre=bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "spike" item.'),
        post=bp.types.NeuState(['V', 'input', 'spike'], help='Pre-synaptic neuron state must have "V" and "input" item.'),
    )

    @bp.integrate
    def int_g(g, _t_):
        return -g / tau_decay
    
    def my_relu(w):
        return w if w > 0 else 0

    def update(ST, _t_, pre, post):
        g = int_g(ST['g'], _t_)
        w = ST['w']
        if pre['spike']:
            g += w
            ST['A_s'] = ST['A_s'] * np.exp((ST['last_spike'] - _t_) / tau_s) + delta_A_s
            ST['A_t'] = ST['A_t'] * np.exp((ST['last_spike'] - _t_) / tau_t)
            w = np.clip(my_relu(ST['w'] - ST['A_t']), w_min, w_max)
            ST['last_spike'] = _t_
        if post['spike']:
            ST['A_s'] = ST['A_s'] * np.exp((ST['last_spike'] - _t_) / tau_s)
            ST['A_t'] = ST['A_t'] * np.exp((ST['last_spike'] - _t_) / tau_t) + delta_A_t
            w = np.clip(my_relu(ST['w'] + ST['A_s']), w_min, w_max)
            ST['last_spike']  =_t_
        ST['w'] = w
        ST['g'] = g


    @bp.delayed
    def output(ST, post):
        I_syn = - g_max * ST['w'] * (post['V'] - E)
        post['input'] += I_syn

    return bp.SynType(name='STDP_AMPA_synapse',
                      requires=requires,
                      steps=(update, output),
                      vector_based=False)
                    
                    
if __name__ == '__main__':
    duration = 550.
    dt = 0.02
    bp.profile.set(backend = "numpy", dt = dt, merge_steps = True, show_code = False)
    LIF_neuron = get_LIF()
    STDP_syn = get_STDP1()

    #build and simulate
    pre = bp.NeuGroup(LIF_neuron, geometry = (3,), monitors = ['V', 'input', 'spike'])
    pre.runner.set_schedule(['input', 'update', 'monitor', 'reset'])
    pre.pars['V_rest'] = -65.
    pre.ST['V'] = -65.
    post = bp.NeuGroup(LIF_neuron, geometry = (3,), monitors = ['V', 'input', 'spike'])
    post.runner.set_schedule(['input', 'update', 'monitor', 'reset'])
    post.pars['V_rest'] = -65.
    post.ST['V'] = -65.

    stdp = bp.SynConn(model = STDP_syn, pre_group = pre, post_group = post, 
                       conn = bp.connect.All2All(), monitors = ['w', 'A_s', 'A_t', 'g'], delay = 10.)
    stdp.ST["A_s"] = 0.
    stdp.ST["A_t"] = 0.
    stdp.ST['w'] = 5.
    stdp.runner.set_schedule(['input', 'update', 'output', 'monitor'])
    
    net = bp.Network(pre, stdp, post)
    
    delta_t = [-20, -15, -10, -8, -6, -4, -3, -2, -1, -0.6, -0.3, -0.2, -0.1, 0, 
               0.1, 0.2, 0.3, 0.6, 1, 2, 3, 4, 6, 8, 10, 15, 20]
    base_t = range(50, 550, 50)
    leng_delta = len(delta_t)
    leng_ext = len(base_t)
    delta_w_list = []
    step = 0
    epoch = leng_delta
    for i in range(leng_delta):
        stdp.ST["A_s"] = 0.
        stdp.ST["A_t"] = 0.
        stdp.ST['w'] = 10.
        
        I_ext_pre = []
        I_ext_post = []
        for j in range(leng_ext):
            I_ext_pre.append(base_t[j])
            I_ext_post.append(base_t[j] + delta_t[i])
    
        current_pre = bp.inputs.spike_current(I_ext_pre, 
                                              bp.profile._dt, 1., duration = duration)
        current_post = bp.inputs.spike_current(I_ext_post, 
                                               bp.profile._dt, 1., duration = duration)
                                               
        net.run(duration = duration, 
                inputs = (
                [stdp, 'pre.spike', current_pre, "="], 
                [stdp, 'post.spike', current_post, "="]
                ), 
                report = False)
        avg_w = np.mean(stdp.mon.w, axis = 1)
        delta_w = 0
        for j in range(leng_ext):
            base = int(I_ext_pre[j]//dt)
            bias = int(I_ext_post[j]//dt)
            if base > bias:
            	deltaw = avg_w[base + 10] - avg_w[bias - 10]
            else:
            	deltaw = avg_w[bias + 10] - avg_w[base - 10]
            delta_w += deltaw
        delta_w /= leng_ext
        delta_w_list.append(delta_w)
        
        step += 1
        print("epoch %2i/%3i, delta_t = %5.1f, delat_w = %f"
              %(step, epoch, delta_t[i], delta_w_list[i]))

        
    ts = net.ts
    fig, gs = bp.visualize.get_figure(1, 1, 6, 8)
    ax = fig.add_subplot(gs[0, 0])
    fig.add_subplot(ax)
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0.))
    plt.scatter(delta_t, delta_w_list, label = 'delta_w-delta_t')
    #plt.xlabel('Delta T')
    #plt.ylabel('Delta synapse weight')
    plt.legend()
    plt.show()
