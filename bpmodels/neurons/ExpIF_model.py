# -*- coding: utf-8 -*-
import numpy as np
import brainpy as bp
import matplotlib.pyplot as plt

## define Exponential Leaky Integrate-and-Fire model
def get_ExpIF(V_rest = -65., V_reset = -68. , V_th = -30., V_T = -59.9, delta_T = 3.48, 
                  R = 10., C = 1., tau = 10., t_refractory = 1.7, noise = 0.):
    '''Exponential Integrate-and-Fire neuron model.
        
    Args:
        V_rest (float): Resting potential.
        V_reset (float): Reset potential after spike.
        V_th (float): Threshold potential of spike.
        V_T (float): Threshold potential of steady/non-steady.
        delta_T (float): Spike slope factor.
        R (float): Membrane Resistance.
        C (float): Membrane Capacitance.
        tau (float): Membrane time constant. Compute by Rm * Cm.
        t_refractory (int): Refractory period length.
        noise (float): noise.   
        
    Returns:
        bp.Neutype: return description of ExpIF model.
    '''
    
    ST = bp.types.NeuState(
        {'V': 0, 'input':0, 'spike':0, 'refractory': 0, 't_last_spike': -1e7}
    )  
    
    @bp.integrate
    def int_V(V, _t_, I_ext):  # integrate u(t)
        return (- ( V - V_rest ) + delta_T * np.exp((V - V_T)/delta_T) + R * I_ext) / tau, noise / tau

    def update(ST, _t_):  
        # update variables
        ST['spike'] = 0
        ST['refractory'] = True if _t_ - ST['t_last_spike'] <= t_refractory else False
        if not ST['refractory']:
            V = int_V(ST['V'], _t_, ST['input'])
            if V >= V_th:
                V = V_reset
                ST['spike'] = 1
                ST['t_last_spike'] = _t_
            ST['V'] = V
    
    def reset(ST):
        ST['input'] = 0.
    
    return bp.NeuType(name = 'ExpIF_neuron', requires = dict(ST=ST), steps = [update, reset], vector_based = False)

if __name__ == '__main__':
    print("version：", bp.__version__)
    ## set global params
    dt = 0.125        # update variables per <dt> ms
    duration = 350.  # simulate duration
    bp.profile.set(backend = "numba", dt = dt, merge_steps = True, show_code = False)
    
    # define neuron type
    Exp_LIF_neuron = get_ExpIF()
    
    # build neuron group
    neu = bp.NeuGroup(Exp_LIF_neuron, geometry = (10, ), monitors = ['V'])
    neu.pars['V_rest'] = np.random.randint(-65, -63, size = (10,))
    neu.pars['tau'] = np.random.randint(5, 10, size = (10,))
    neu.pars['noise'] = 1.
    
    # create input
    current, pos_dur = bp.inputs.constant_current([(0.30, duration)])
    
    # simulate
    neu.run(duration = pos_dur, inputs = ["ST.input", current], report = True)  
    #simulate for 100 ms. Give external input = current

    # paint
    ts = neu.mon.ts
    fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
    fig.add_subplot(gs[0, 0])
    plt.plot(ts, neu.mon.V[:, 0], label = f'neuron No.{0}, V_rest = {neu.pars.get("V_rest")[0]}mV, tau = {neu.pars.get("tau")[0]}ms.')
    plt.plot(ts, neu.mon.V[:, 6], label = f'neuron No.{6}, V_rest = {neu.pars.get("V_rest")[6]}mV, tau = {neu.pars.get("tau")[6]}ms.')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, ts[-1] + 0.1)
    plt.legend()
    plt.show()
