import numpy as np
import npbrain as nb
import matplotlib.pyplot as plt
import pdb

## set global params
# set rand seed
np.random.seed(42) 
# set params
dt = 0.02        # update variables per <dt> ms
duration = 100.  # simulate duration
nb.profile.set(backend = "numpy", dt = dt, merge_ing = True)

## define Leaky Integrate-and-Fire model
def LIF_model(Vr = 0, Vth = 20, Rm = 1, Cm = 10, tau_m = 10, refTime = 5, noise = 0.):
    ST = nb.types.NeuState(
        {'Vm': 0, 'refState': 0, 'input':0, 'spikeCnt':0}, 
        help = '''
            LIF neuron model.
            Vm: voltage of membrane.
            refState: refractory state
            input: external input, from stimulus and other synapses
            spikeCnt: total spike cnt (record to compute firing rate)
        '''
    )
    
    @nb.integrate(noise = noise)
    def int_v(V, _t_, input):  # integrate u(t)
        return (- ( V - Vr ) + Rm * input) / tau_m

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
    
    return nb.NeuType(name = 'LIF_neuron', requires = dict(ST=ST), steps = update, vector_based = False)

neu = nb.NeuGroup(LIF_model, geometry = (10, ), monitors = ['Vm'],
                  pars_update = {
                  'Vr': np.random.randint(0, 2, size = (10,)),
                  'tau_m': np.random.randint(5, 10, size = (10,)),
                  'noise': 1.
                 })  #create a neuron group with 10 neurons.
net = nb.Network(neu)

pdb.set_trace()

net.run(duration = duration, inputs = [neu, "ST.input", 26.], report = True)  
#simulate for 100 ms. Give external input = [receiver, field name, strength]

pdb.set_trace()

#paint
ts = net.ts
fig, gs = nb.visualize.get_figure(1, 1, 4, 8)

fig.add_subplot(gs[0, 0])
plt.plot(ts, neu.mon.Vm[:, 0], label = f'neuron No.{0}, Vr = {neu.pars_update["Vr"][0]}mV, tau_m = {neu.pars_update["tau_m"][0]}ms.')
plt.plot(ts, neu.mon.Vm[:, 1], label = f'neuron No.{6}, Vr = {neu.pars_update["Vr"][6]}mV, tau_m = {neu.pars_update["tau_m"][6]}ms.')
#paint Vm-t plot of 1st and 2nd neuron
plt.ylabel('Membrane potential')
plt.xlim(-0.1, net._run_time + 0.1)
plt.legend()
plt.xlabel('Time (ms)')

plt.show()