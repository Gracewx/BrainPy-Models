import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt
from bpmodels.learning_rules import get_BCM


def rate_neuron():
    ST = bp.types.NeuState(['r', 'input'])

    def g(r, I):
        r += I
        return r

    def update(ST):
        ST['r'] = g(ST['r'], ST['input'])
    
    def reset(ST):
        ST['input'] = 0.

    return bp.NeuType(name='rate', steps=[update, reset], ST=ST, mode='vector')




w_max = 2.

bp.profile.set(dt=.1)

n_post = 1

neuron = rate_neuron()
post = bp.NeuGroup(neuron, n_post, monitors=['r'])
pre = bp.NeuGroup(neuron, 20, monitors=['r'])

#mode = 'matrix'
mode = 'vector'
print(mode)

bcm1 = get_BCM(learning_rate=0.005, w_max=w_max, mode=mode)
bcm = bp.SynConn(model=bcm1, pre_group=pre, post_group=post,
                    conn=bp.connect.All2All(), 
                    monitors=['w', 'dwdt'],
                    delay = 0)
bcm.r_th = np.zeros(n_post)
bcm.post_r = np.zeros(n_post)
bcm.sum_post_r = np.zeros(n_post)


net = bp.Network(pre, bcm, post)

# group selection

group1, duration = bp.inputs.constant_current(([1.5, 1], [0, 1])*20)
group2, duration = bp.inputs.constant_current(([0, 1], [1., 1])*20)

input_r = np.vstack((
                    (group1,)*10, (group2,)*10
                    ))


net.run(duration, inputs=(pre, 'ST.r', input_r.T, "="), report=True)

if mode == 'matrix':
    w1 = np.mean(bcm.mon.w[:,:10,0], 1)
    w2 = np.mean(bcm.mon.w[:,10:,0], 1)
else:
    w1 = np.mean(bcm.mon.w[:,:10], 1)
    w2 = np.mean(bcm.mon.w[:,10:], 1)

r1 = np.mean(pre.mon.r[:, :10], 1)
r2 = np.mean(pre.mon.r[:, 10:], 1)
post_r = np.mean(post.mon.r[:, :], 1)

fig, gs = bp.visualize.get_figure(3, 1, 2, 6)
fig.add_subplot(gs[2, 0], xlim=(0, duration), ylim=(0, w_max))
plt.plot(net.ts, w1, 'b', label='group1')
plt.plot(net.ts, w2, 'r', label='group2')
plt.title("weights")
plt.ylabel("weights")
plt.xlabel("t")
plt.legend()

fig.add_subplot(gs[0, 0], xlim=(0, duration))
plt.plot(net.ts, r1, 'b', label='group1')
plt.plot(net.ts, r2, 'r', label='group2')
plt.title("inputs")
plt.ylabel("firing rate")
plt.xlabel("t")
plt.legend()

fig.add_subplot(gs[1, 0], xlim=(0, duration))
plt.plot(net.ts, post_r, 'g', label='post_r')
plt.title("response")
plt.ylabel("firing rate")
plt.xlabel("t")
plt.legend()

plt.show()
