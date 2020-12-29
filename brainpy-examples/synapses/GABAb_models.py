# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import brainpy as bp
import numpy as np
import bpmodels
from bpmodels.neurons import get_LIF

duration = 500.
dt = 0.02
bp.profile.set(jit=True, dt=dt, merge_steps=True, show_code=False)
LIF_neuron = get_LIF()
GABAa_syn = bpmodels.synapses.get_GABAb1(mode='vector')

# build and simulate gabaa net
pre = bp.NeuGroup(LIF_neuron, geometry=(10,), monitors=['V', 'input', 'spike'])
pre.runner.set_schedule(['input', 'update', 'monitor', 'reset'])
pre.pars['V_rest'] = -65.
pre.ST['V'] = -65.
post = bp.NeuGroup(LIF_neuron, geometry=(10,), monitors=['V', 'input', 'spike'])
post.runner.set_schedule(['input', 'update', 'monitor', 'reset'])
post.pars['V_rest'] = -65.
post.ST['V'] = -65.

gabab = bp.SynConn(model=GABAa_syn, pre_group=pre, post_group=post,
                   conn=bp.connect.All2All(), monitors=['g'], delay=10.)

net = bp.Network(pre, gabab, post)

current = bp.inputs.spike_current([5, 10, 15, 20],
                                  bp.profile._dt, 1., duration=duration)
net.run(duration=duration, inputs=[gabab, 'pre.spike', current, "="], report=True)

# paint gabaa
ts = net.ts
fig, gs = bp.visualize.get_figure(2, 1, 5, 6)

fig.add_subplot(gs[0, 0])
plt.plot(ts, gabab.mon.g[:, 0], label='g')
plt.legend()

fig.add_subplot(gs[1, 0])
plt.plot(ts, post.mon.V[:, 0], label='post.V')
plt.legend()

plt.show()
