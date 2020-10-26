import numpy as np
import brainpy as bp
import matplotlib.pyplot as plt

print("version：", bp.__version__)
## set global params
dt = 0.02        # update variables per <dt> ms
duration = 100.  # simulate duration
bp.profile.set(backend = "numba", dt = dt, merge_steps = True)

## define Leaky Integrate-and-Fire model
def LIF_model(Vr = 0., Vth = 20., Rm = 1., Cm = 10., tau_m = 10., refTime = 5., noise = 0.):
    ST = bp.types.NeuState(
        {'Vm': 0, 'refState': 0, 'input':0, 'spikeCnt':0}
    )  
    '''
    LIF neuron model.
    Vm: voltage of membrane.
    refState: refractory state.
    input: external input, from stimulus and other synapses.
    spikeCnt: total spike cnt (record to compute firing rate).
    '''
    
    @bp.integrate
    def int_v(V, _t_, I_syn):  # integrate u(t)
        return (- ( V - Vr ) + Rm * I_syn) / tau_m, noise / tau_m

    def update(ST, _t_):  
        # update variables
        refPeriod = refTime // dt  #refractory
        if ST['refState'] <= 0:
            V = int_v(ST['Vm'], _t_, ST['input'])
            if V >= Vth:
                V = Vr
                ST['refState'] = refPeriod
                ST['spikeCnt'] += 1
            ST['Vm'] = V
        else:
            ST['refState'] -= 1
        ST['input'] = 0.  #ST['input'] is current input (only valid for current step, need reset each step)
    
    return bp.NeuType(name = 'LIF_neuron', requires = dict(ST=ST), steps = update, vector_based = False)

LIF_neuron = LIF_model(noise = 1.)

neu = bp.NeuGroup(LIF_neuron, geometry = (10, ), monitors = ['Vm'],
                  pars_update = {
                  'Vr': np.random.randint(0, 2, size = (10,)),
                  'tau_m': np.random.randint(5, 10, size = (10,))  ,
                  'noise': 1.
                 })  #create a neuron group with 10 neurons.
net = bp.Network(neu)

net.run(duration = duration, inputs = [neu, "ST.input", 26.], report = True)  
#simulate for 100 ms. Give external input = [receiver, field name, strength]

#paint
ts = net.ts
fig, gs = bp.visualize.get_figure(1, 1, 4, 8)

fig.add_subplot(gs[0, 0])
plt.plot(ts, neu.mon.Vm[:, 0], label = f'neuron No.{0}, Vr = {neu.pars.get("Vr")[0]}mV, tau_m = {neu.pars.get("tau_m")[0]}ms.')
plt.plot(ts, neu.mon.Vm[:, 6], label = f'neuron No.{6}, Vr = {neu.pars.get("Vr")[6]}mV, tau_m = {neu.pars.get("tau_m")[6]}ms.')
#paint Vm-t plot of 1st and 2nd neuron
plt.ylabel('Membrane potential')
plt.xlim(-0.1, net.t_end - net.t_start + 0.1)
plt.legend()
plt.xlabel('Time (ms)')

plt.show()