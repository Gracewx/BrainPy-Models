# -*- coding: utf-8 -*-
import numpy as np
import brainpy as bp
import matplotlib.pyplot as plt

## define Leaky Integrate-and-Fire model
def get_LIF(Vr = 0., Vreset = -5.,  Vth = 20., Rm = 1., Cm = 10., tau_m = 10., refPeriod = 5.//0.02, noise = 0.):
    ST = bp.types.NeuState(
        {'Vm': 0, 'refState': 0, 'input':0, 'spikeCnt':0, 'isFire': 0}
    )  
    '''Leaky Integrate-and-Fire neuron model.
    
    ..math::
        \\tau_m \\frac{d V_m}{d t} &= - ( V_m - V_{rest}) + RI(t)
    
    
    Args:
        Vrest (float): Resting potential.
        Vreset (float): Reset potential after spike.
        Vth (float): Threshold potential of spike.
        Rm (float): Membrane Resistance.
        Cm (float): Membrane Capacitance.
        tau_m (float): Membrane time constant. Compute by Rm * Cm.
        refPeriod (int): Refractory period length.
        noise (float): noise.   
        
    Returns:
        bp.Neutype: return description of LIF model.
    '''
    
    @bp.integrate
    def int_v(V, _t_, I_syn):  # integrate u(t)
        return (- ( V - Vr ) + Rm * I_syn) / tau_m, noise / tau_m

    def update(ST, _t_):  
        # update variables
        ST['isFire'] = 0
        if ST['refState'] <= 0:
            V = int_v(ST['Vm'], _t_, ST['input'])
            if V >= Vth:
                V = Vreset
                ST['refState'] = refPeriod
                ST['spikeCnt'] += 1
                ST['isFire'] = 1
            ST['Vm'] = V
        else:
            ST['refState'] -= 1
    
    def reset(ST):
        ST['input'] = 0.  #ST['input'] is current input (only valid for current step, need reset each step)
    
    return bp.NeuType(name = 'LIF_neuron', requires = dict(ST=ST), steps = [update, reset], vector_based = False)
    
if __name__ == '__main__':
    print("version：", bp.__version__)
    ## set global params
    dt = 0.02        # update variables per <dt> ms
    duration = 100.  # simulate duration
    bp.profile.set(backend = "numba", dt = dt, merge_steps = True)
    
    # define neuron type
    LIF_neuron = get_LIF(noise = 1.)
    
    # build neuron group
    neu = bp.NeuGroup(LIF_neuron, geometry = (10, ), monitors = ['Vm'])  
    neu.pars['Vr'] = np.random.randint(0, 2, size = (10,))
    neu.pars['tau_m'] = np.random.randint(5, 10, size = (10,))
    neu.pars['noise'] = 1.
    neu.runner.set_schedule(['input', 'update', 'monitor', 'reset'])

    #simulate
    neu.run(duration = duration, inputs = ["ST.input", 26.], report = True)  
    #simulate for 100 ms. Give external input = [receiver, field name, strength]

    #paint
    ts = neu.mon.ts
    fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
    fig.add_subplot(gs[0, 0])
    plt.plot(ts, neu.mon.Vm[:, 0], label = f'neuron No.{0}, Vr = {neu.pars.get("Vr")[0]}mV, tau_m = {neu.pars.get("tau_m")[0]}ms.')
    plt.plot(ts, neu.mon.Vm[:, 6], label = f'neuron No.{6}, Vr = {neu.pars.get("Vr")[6]}mV, tau_m = {neu.pars.get("tau_m")[6]}ms.')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, ts[-1] + 0.1)
    plt.legend()
    plt.show()
