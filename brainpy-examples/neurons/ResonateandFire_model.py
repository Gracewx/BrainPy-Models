# -*- coding: utf-8 -*-
import brainpy as bp
import bpmodels
import numpy as np
import matplotlib.pyplot as plt

print("version：", bp.__version__)
## set global params
dt = 0.002  # update variables per <dt> ms
duration = 20.  # simulate duration
bp.profile.set(jit=True, dt=dt, merge_steps=True, show_code=False)

# define neuron type
RF_neuron = bpmodels.neurons.get_ResonateandFire()

# build neuron group
neu = bp.NeuGroup(RF_neuron, geometry=(10,), monitors=['x', 'V', 'spike'])
neu.runner.set_schedule(['input', 'update', 'monitor', 'reset'])

# create input
current = bp.inputs.spike_current([0.1],
                                  bp.profile._dt, -2., duration=duration)

# simulate
neu.run(duration=duration, inputs=["ST.input", current], report=True)
# simulate for 100 ms. Give external input = current

# paint
ts = neu.mon.ts
fig, gs = bp.visualize.get_figure(1, 2, 4, 8)
fig.add_subplot(gs[0, 0])
plt.scatter(neu.mon.x[:, 0], neu.mon.V[:, 0], label = "V-x plot")
plt.xlabel('x')
plt.ylabel('V')
plt.legend()

fig.add_subplot(gs[0, 1])
plt.plot(ts, neu.mon.V[:, 0], label = "V-t plot")
plt.plot(ts, neu.mon.V[:, 0], label = "V-t plot")
plt.xlabel('Time (ms)')
plt.ylabel('V')
plt.legend()
plt.show()
