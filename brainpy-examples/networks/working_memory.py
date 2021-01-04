import brainpy as bp
import brainpy.numpy as np
import bpmodels
import matplotlib.pyplot as plt

# set params
dt=0.02
bp.profile.set(backend='numba',
               device='cpu',
               dt=dt,
               merge_steps=True)

base_N_E = 2048
base_N_I = 512
scale = 8
N_E = base_N_E//scale
N_I = base_N_I//scale

## set neuron params
### E-neurons/pyramidal cells
C_E = 0.5
g_E = 25.
V_rest_E = -70.
V_reset_E = -60.
V_th_E = -50.
tau_E = 2.
R_E = tau_E/C_E = 1/g_E #???
### I-neurons/interneurons
C_I = 0.2
g_I = 20.
V_rest_I = -70.
V_reset_I = -60.
V_th_I = -50.
tau_I = 1.
R_I = tau_I/C_I = 1/g_I #???

## set input params
possion_frequency = 1800 #or 1000*1.8 #be precise, what is that?
g_max_ext_E = 3.1  #AMPA
g_max_ext_I = 2.38 #AMPA

## set synapse params
### AMPA
tau_AMPA = 2.
E_AMPA = 0.
### GABAa
tau_GABAa = 10.
E_GABAa = -70.
### NMDA
tau_delay_NMDA = 100.
tau_rise_NMDA = 2.
cc_Mg_NMDA = 1.
alpha_NMDA = 0.062
beta_NMDA = 3.57
a_NMDA = 0.5 #kHz #TODO: check lianggang
E_NMDA = 0.
g_max_EE_NMDA = 0.381
g_max_EI_NMDA = 0.292
g_max_IE_NMDA = 1.336
g_max_II_NMDA = 1.024

delta_EE = 18du #TODO: check danwei
J_plus_EE = 1.62
JEE =  #TODO: build connection
JEI = 
JIE = 
JII = 

# ===============
# fake_neuron
# ===============

def get_fake_neuron():

    ST = bp.types.NeuState({'spike': 0})
    
    def reset(ST, _t_):
        ST['spike'] = 0.
    
    return bp.NeuType(name='fake_neuron',
                      requires=dict(ST=ST),
                      steps=reset,
                      mode='scalar')

# get neu & syn type
fake_neuron = get_fake_neuron()
LIF_neuron = bpmodels.neurons.get_LIF()
AMPA_synapse = bpmodels.synapses.get_AMPA()
GABAa_synapse = bpmodels.synapses.get_GABAa()

# build neuron groups & synapse connections
input_neu = bp.NeuGroup(fake_neuron, geometry = (N_E + N_I, ), monitors = [])
excit_neu = bp.NeuGroup(LIF_neuron, geometry = (N_E, ), monitors = ['spike'])
inhib_neu = bp.NeuGroup(LIF_neuron, genmetry = (N_I, ), monitors = ['spike'])
ext2excit_syn = bp.SynConn(AMPA_synapse, pre_group = input_neu[:N_E], post_group = excit_neu,
                           conn = bp.connect.One2One(), monitors = [], delay = 10.)
ext2inhib_syn = bp.SynConn(AMPA_synapse, pre_group = input_neu[N_E:], post_group = inhib_neu,
                           conn = bp.connect.One2One(), monitors = [], delay = 10.)
#TODO: check how to set delay time?
excit2excit_syn = bp.SynConn()

#TODO: set neuron param here
#TODO: set AMPA param here
#TODO: set GABAa param here
background_input = #TODO:set 1800Hz input here
