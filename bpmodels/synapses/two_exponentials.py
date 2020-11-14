# -*- coding: utf-8 -*-
import brainpy as bp
import brainpy.numpy as np

def get_two_exponentials(g_max=0.2, E=-60., tau_d=0.8, tau_r=5.):
    '''
        two_exponentials synapse model.

        Args:
            g_max (float): The peak conductance change in µmho (µS).
            E (float): The reversal potential for the synaptic current.
            tau_r (float): The time to peak of the conductance change.
            tau_d (float): The decay time of the synapse.

        Returns:
            bp.Neutype: return description of two_exponentials synapse model.
    '''

    requires = {
        'ST': bp.types.SynState(['g'],help='The conductance defined by two_exponentials function.'),
        'pre': bp.types.NeuState(['V'], help='pre-synaptic neuron state must have "V"'),
        'post': bp.types.NeuState(['input', 'V'], help='post-synaptic neuron state must include "input" and "V"')
    }

    def update(ST, _t_):
            ST['g'] = g_max * (np.exp(-_t_/ tau_d)+np.exp(-_t_/ tau_r))

    @bp.delayed
    def output(ST, _t_, pre, post):
        I_syn = ST['g'] * (pre['V'] - E)
        post['input'] += I_syn

    return bp.SynType(name='two_exponentials_synapse',
                      requires=requires,
                      steps=(update, output),
                      vector_based=False)