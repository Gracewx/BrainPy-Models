from bpmodels.neurons import HH
import brainpy as bp

duration = 80.

# use "profile" to make global settings.
bp.profile.set(backend='numpy', dt=0.02, numerical_method='milstein', merge_steps=True)

# Example 1
# quickly run a simulation by calling HH.simulate
(ts, V, m, h, n) = HH.simulate(input_current = 5., duration = duration, geometry=1)

# Example 2
# get neuron by HH.get_neuron, then run simulation
HH_neuron = HH.get_neuron(geometry=1, noise=0.)
HH_neuron.run(duration=duration, inputs=['ST.input', 5.], report=True)

ts = HH_neuron.mon.ts
V = HH_neuron.mon.V[:, 0]
m = HH_neuron.mon.m[:, 0]
h = HH_neuron.mon.h[:, 0]
n = HH_neuron.mon.n[:, 0]

# Example 3
# call HH.define_HH to create a neuron type.
HH = HH.define_HH(noise = 0.)
HH_neuron = bp.NeuGroup(HH, geometry=1, 
                    monitors=['spike', 'V', 'm', 'h', 'n'])

HH_neuron.run(duration=duration, inputs=['ST.input', 5.], report=True)

ts = HH_neuron.mon.ts
V = HH_neuron.mon.V[:, 0]
m = HH_neuron.mon.m[:, 0]
h = HH_neuron.mon.h[:, 0]
n = HH_neuron.mon.n[:, 0]


# Visualization

import matplotlib.pyplot as plt

fig, gs = bp.visualize.get_figure(2, 1, 3, 8)

# plot membrane potential
fig.add_subplot(gs[0, 0])
plt.plot(ts, V)
plt.ylabel('Membrane potential (mV)')
plt.xlim(-0.1, duration + 0.1)
plt.title('Membrane potential')

# plot gate variables
fig.add_subplot(gs[1, 0])
plt.plot(ts, m, label='m')
plt.plot(ts, h, label='h')
plt.plot(ts, n, label='n')
plt.legend()
plt.xlim(-0.1, duration + 0.1)
plt.xlabel('Time (ms)')
plt.ylabel('gate variables')
plt.title('gate variables')

plt.show()