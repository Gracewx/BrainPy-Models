import brainpy as bp
import brainpy.numpy as np
import bpmodels
import matplotlib.pyplot as plt

def run_neuron(neu_type, I_ext=10., duration=100., monitors=['V'], plot=False):
    neuron = bp.NeuGroup(neu_type, 1, monitors=monitors)

    #neuron.ST['V'] = -87.

    neuron.run(duration=duration, inputs=["ST.input", I_ext])
    print(neu_type, 'done!')

    if plot:
        # read time sequence
        ts = neuron.mon.ts
        fig, gs = bp.visualize.get_figure(2, 1, 3, 6)

        V = neuron.mon.V[:, 0]

        # plot input current
        fig.add_subplot(gs[0, 0])
        try:
            plt.plot(ts, I_ext, 'r')
        except:
            plt.plot(ts, I_ext*np.ones(len(ts)), 'r')
        else:
            pass
        plt.ylabel('Input Current')
        plt.xlim(-0.1, duration + 0.1)
        plt.xlabel('Time (ms)')
        plt.title('Input Current')

        # plot membrane potential
        fig.add_subplot(gs[1, 0])
        plt.plot(ts, V, label='V')
        plt.ylabel('Membrane potential')
        plt.xlabel('Time (ms)')
        plt.xlim(-0.1, duration + 0.1)
        #plt.ylim(-95., 40.)
        plt.title('Membrane potential')
        plt.legend()

        '''
        fig.add_subplot(gs[2, 0])
        plt.plot(ts, neuron.mon.u[:, 0], label='u')
        plt.xlim(-0.1, duration + 0.1)
        plt.ylabel('Recovery variable')
        plt.xlabel('Time (ms)')
        plt.title('Recovery variable')
        plt.legend()
        '''

        plt.show()

def run_net(neu_type, syn_type, duration=350., num=1, plot=True, report=False):
    # print('Now testing', syn_type)
    pre = bp.NeuGroup(neu_type, 1, monitors=['spike', 'V'])
    post = bp.NeuGroup(neu_type, 1, monitors=['input', 'V'])
    pre.ST['V'] = -65.
    post.ST['V'] = -65.

    syn = bp.SynConn(model=syn_type, pre_group=pre, post_group=post, conn=bp.connect.All2All(),
                     delay=1.5, monitors=['g'])

    post.runner.set_schedule(['input','update','monitor', 'reset'])
    syn.runner.set_schedule(['input', 'update', 'output', 'monitor'])

    net = bp.Network(pre, syn, post)

    
    Iext = bp.inputs.spike_current(
        [10, 15, 20, 25, 210, 310, 410], bp.profile._dt, 1., duration=duration)
    net.run(duration, inputs=(syn, 'pre.spike', Iext, '='), report=False)
    

    #net.run(duration, inputs=(pre, 'ST.input', 100.), report=report)

    print(syn_type, 'done!')

    if plot:
        fig, gs = bp.visualize.get_figure(3, 1, 2, 6)

        fig.add_subplot(gs[2, 0])
        #plt.plot(net.ts, pre.mon.spike[:, 0], 'r', label = 'spike')
        plt.plot(net.ts, Iext, 'r', label = 'spike')
        plt.plot(net.ts, syn.mon.g[:, 0], 'b', label='g')
        plt.ylim(0, 0.5)
        plt.legend()

        fig.add_subplot(gs[0, 0])
        plt.plot(net.ts, pre.mon.V[:, 0], label='pre_V')
        #plt.plot(net.ts, post.mon.V[:, 0], label='post_V')
        #plt.ylim(-95., 40.)
        plt.legend()

        fig.add_subplot(gs[1, 0])
        plt.plot(net.ts, post.mon.input[:, 0], label='post_input')
        plt.ylim(0, 2)
        plt.legend()
        plt.show()

def plot_lr(neu, syn_type, dt=.1):
    pre = bp.NeuGroup(neu, 1)
    post = bp.NeuGroup(neu, 1)

    syn = bp.SynConn(syn_type, pre_group=pre, post_group=post,
                     conn=bp.connect.All2All(), monitors=['A_s', 'A_t', 'w', 'g'])
    net = bp.Network(pre, post, syn)

    # input
    duration = 100.
    pre_ts = np.arange(5, duration, 10)
    dts = np.arange(pre_ts.size)

    post_ts = pre_ts + dts
    pre_spikes = bp.inputs.spike_current(
        pre_ts.tolist(), dt, 1., duration=duration)
    post_spikes = bp.inputs.spike_current(
        post_ts.tolist(), dt, 1., duration=duration)
    net.run(duration=duration, inputs=[(syn, 'pre.spike', pre_spikes),
                                        (syn, 'post.spike', post_spikes)])

    # visualize
    fig, gs = bp.visualize.get_figure(3, 1, 2, 9)

    fig.add_subplot(gs[0, 0])
    plt.plot(net.ts, syn.mon.A_s[:, 0], label='pre_spike')
    plt.plot(net.ts, syn.mon.A_t[:, 0], label='post_spike')
    plt.xlabel('Time (ms)')
    plt.title('Spike traces')
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(net.ts, syn.mon.w[:, 0])
    plt.ylabel('weights')
    plt.xlabel('Time (ms)')
    plt.title('synaptic weights')

    fig.add_subplot(gs[2, 0])
    plt.plot(net.ts, syn.mon.g[:, 0])
    plt.ylabel('s')
    plt.xlabel('Time (ms)')
    plt.title('synaptic conductances')

    plt.show()

def get_current(amplitude=30, duration=300, mode='step'):
    a = duration/6
    if mode == 'step':
        (I, duration) = bp.inputs.constant_current(
                                [(0, a), (amplitude, duration-2*a), (0, a)])
    elif mode == 'ramp':
        ramp_I = bp.inputs.ramp_current(c_start = 0.,
                                    c_end = a,
                                    duration = duration,
                                    t_start = a,
                                    t_end = duration-a)

    return I

def AdEx_patterns():
    patterns = dict(tonic = [20, 0.0, 30.0, .6, -55],
                    adapting = [20, 0., 100, .05, -55],
                    init_bursting = [5., .5, 100, .07, -51],
                    bursting = [5., -.5, 100, .07, -46],
                    irregular = [9.9, -.5, 100, .07, -46],
                    transient = [10, 1., 100, .1, -60],
                    delayed = [5, -1, 100, .1, -60])

    for i in patterns:
        print(i)
        i = patterns[i]
        neu = bpmodels.neurons.get_AdExIF(tau=i[0],
                                            a=i[1],
                                            tau_w=i[2],
                                            b=i[3],
                                            V_reset=-70,
                                            V_rest=-70,
                                            R=.5,
                                            delta_T=2.,
                                            V_th=-i[4],
                                            V_T=-50)
        run_neuron(neu, I_ext=65, duration=300., plot=True)

def izh_patterns(fire_type='tonic spiking'):
    izh = bpmodels.neurons.get_Izhikevich(type=fire_type)
    # izh = bpmodels.neurons.get_Izhikevich(a=0.02, b=0.2, c=-65, d = 8)
    duration=300
    I = get_current(duration=duration)
    run_neuron(izh, I, duration, monitors=['V', 'u'])

def test_all_neurons(I_ext=30., duration=100., plot=False):
    HH = bpmodels.neurons.get_HH()
    run_neuron(HH, I_ext=I_ext, duration=duration, plot=plot)

    izh = bpmodels.neurons.get_Izhikevich()
    run_neuron(izh, I_ext=I_ext, duration=duration, plot=plot)

    LIF = bpmodels.neurons.get_LIF()
    run_neuron(LIF, I_ext=I_ext, duration=duration, plot=plot)

    expIF = bpmodels.neurons.get_ExpIF()
    run_neuron(expIF, I_ext=3., duration=100., plot=plot)

    aEIF = bpmodels.neurons.get_AdExIF()
    run_neuron(aEIF, I_ext=3., duration=100., plot=plot)

    QIF = bpmodels.neurons.get_QuaIF()
    run_neuron(QIF, I_ext=50., duration=100., plot=plot)

    aQIF = bpmodels.neurons.get_AdQuaIF()
    run_neuron(aQIF, I_ext=20., duration=100., plot=plot)

    genIF = bpmodels.neurons.get_GeneralizedIF()
    run_neuron(genIF, I_ext=I_ext, duration=duration, plot=plot)

    ML = bpmodels.neurons.get_MorrisLecar()
    run_neuron(ML, I_ext=I_ext, duration=duration, plot=plot)

    hr = bpmodels.neurons.get_HindmarshRose()
    run_neuron(hr, I_ext=I_ext, duration=duration, plot=plot)

    wc = bpmodels.neurons.get_WilsonCowan()
    run_neuron(wc, I_ext=I_ext, duration=duration, plot=plot)

def test_all_syns():
    neu = bpmodels.neurons.get_LIF(V_reset= -65., V_rest=-65.)
      
    alpha = bpmodels.synapses.get_alpha()
    run_net(neu, alpha, num=1)

    ampa1 = bpmodels.synapses.get_AMPA1()
    run_net(neu, ampa1)

    ampa2 = bpmodels.synapses.get_AMPA2()
    run_net(neu, ampa2)

    expo = bpmodels.synapses.get_exponential()
    run_net(neu, expo, num=1)
    
    gabaa1 = bpmodels.synapses.get_GABAa1()
    run_net(neu, gabaa1, num=1)

    gabaa2 = bpmodels.synapses.get_GABAa2()
    run_net(neu, gabaa2, num=1)

    gabab1 = bpmodels.synapses.get_GABAb1()
    run_net(neu, gabab1, num=1)

    gabab2 = bpmodels.synapses.get_GABAb2()
    run_net(neu, gabab2, num=1)
    
    nmda = bpmodels.synapses.get_NMDA()
    run_net(neu, nmda, num=1)

    stp = bpmodels.synapses.get_STP()
    run_net(neu, stp, num=1)

    expo2 = bpmodels.synapses.get_two_exponentials()
    run_net(neu, expo2, num=1)

def test_modes():
    neu = bpmodels.neurons.get_LIF(V_reset= -65., V_rest=-65.)

    for mode in ['scalar', 'vector', 'matrix']:
        print(mode)
        syn = bpmodels.synapses.get_STP(mode=mode)
        run_net(neu, syn, report=True)


if __name__ == "__main__":
    bp.profile.set(backend="numba", device='cpu', dt=.1, merge_steps=True)
    
    # test_all_neurons()
    # test_all_syns()
    
    # please modify "get_XX" inside the definition of the function
    # test_modes()

    # izh_patterns()
    #AdEx_patterns()

    #neu = bpmodels.neurons.get_AdQuaIF()
    #run_neuron(neu, I_ext=20., duration=100., plot=True)
    
    #neu = bpmodels.neurons.get_LIF(V_reset= -65., V_rest=-65.) 
    #syn = bpmodels.synapses.get_alpha()
    #run_net(neu, syn)

    '''    
    # test learning rule

    stdp = bpmodels.learning_rules.get_STDP1()
    plot_lr(neu, stdp)
    '''