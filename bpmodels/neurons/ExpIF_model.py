# -*- coding: utf-8 -*-
import numpy as np
import brainpy as bp
import matplotlib.pyplot as plt

## define Exponential Leaky Integrate-and-Fire model
def get_ExpIF(Vrest = -65., Vreset = -68. , Vth = -30., VT = -59.9, delta_T = 3.48, 
                  Rm = 10., Cm = 1., tau_m = 10., refTime = 1.7, noise = 0.):
    ST = bp.types.NeuState(
        {'Vm': -65., 'refState': 0, 'input':0, 'isFire':0, 'spikeCnt':0}
    )  
    '''Exponential Integrate-and-Fire neuron model.
    
    ..math::
        \\tau_m \\frac{d V_m}{d t} &= - ( V_m - V_{rest}) + \\varDelta_T e^{\\frac{V_m-V_{rest}{\\varDelta_T}}} + RI(t)
        
    Args:
        Vrest (float): Resting potential.
        Vreset (float): Reset potential after spike.
        Vth (float): Threshold potential of spike.
        VT (float): Threshold potential of steady/non-steady.
        delta_T (float): Spike slope factor.
        Rm (float): Membrane Resistance.
        Cm (float): Membrane Capacitance.
        tau_m (float): Membrane time constant. Compute by Rm * Cm.
        refPeriod (int): Refractory period length.
        noise (float): noise.   
        
    Returns:
        bp.Neutype: return description of ExpIF model.
    '''
    
    @bp.integrate
    def int_v(V, _t_, I_syn):  # integrate u(t)
        return (- ( V - Vrest ) + delta_T * exp((V - VT)/delta_T) + Rm * I_syn) / tau_m, noise / tau_m

    def update(ST, _t_):  
        # update variables
        refPeriod = refTime // dt  #refractory
        ST['isFire'] = 0
        if ST['refState'] <= 0:
            V = int_v(ST['Vm'], _t_, ST['input'])
            #print(V, Vrest, delta_T, VT, Rm, ST['input'], tau_m)
            if V >= Vth:
                V = Vreset
                ST['refState'] = refPeriod
                ST['spikeCnt'] += 1
                ST['isFire'] = 1
            ST['Vm'] = V
        else:
            ST['refState'] -= 1
        ST['input'] = 0.  #ST['input'] is current input (only valid for current step, need reset each step)
    
    return bp.NeuType(name = 'Exp_LIF_neuron', requires = dict(ST=ST), steps = update, vector_based = False)

if __name__ == '__main__':
    print("version：", bp.__version__)
    ## set global params
    dt = 0.125        # update variables per <dt> ms
    duration = 350.  # simulate duration
    bp.profile.set(backend = "numba", dt = dt, merge_steps = True, show_code = False)
    
    # define neuron type
    Exp_LIF_neuron = get_ExpIF()
    
    # build neuron group
    neu = bp.NeuGroup(Exp_LIF_neuron, geometry = (10, ), monitors = ['Vm'])
    neu.pars['Vrest'] = np.random.randint(-65, -63, size = (10,))
    neu.pars['tau_m'] = np.random.randint(5, 10, size = (10,))
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
    plt.plot(ts, neu.mon.Vm[:, 0], label = f'neuron No.{0}, Vr = {neu.pars.get("Vrest")[0]}mV, tau_m = {neu.pars.get("tau_m")[0]}ms.')
    plt.plot(ts, neu.mon.Vm[:, 6], label = f'neuron No.{6}, Vr = {neu.pars.get("Vrest")[6]}mV, tau_m = {neu.pars.get("tau_m")[6]}ms.')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, ts[-1] + 0.1)
    plt.legend()
    plt.show()
