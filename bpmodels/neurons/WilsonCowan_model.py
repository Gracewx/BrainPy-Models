import brainpy as bp
import numpy as np
import sys

def get_WilsonCowan(c1 = 12., c2 = 4., c3 = 13., c4 = 11.,
                    k_e = 1., k_i = 1.,tau_e = 1., tau_i = 1., r_e = 1., r_i = 1.,
                    slope_e = 1.2, slope_i = 1., theta_e = 2.8, theta_i = 4., mode='scalar'):
    """
    Wilson-Cowan firing rate neuron model.

    Each neuron refers to a column of neurons, including excitatory and 
    inhibitory neurons.

    .. math::
        
        &\\tau_e\\frac{d a_e(t)}{d t} = 
            - a_e(t) + (k_e - r_e * a_e(t)) * 
                        \\mathcal{S}_e(c_1 a_e(t) - c_2 a_i(t) + I_{ext_e}(t))
        
        &\\tau_i\\frac{d a_i(t)}{d t} = 
            - a_i(t) + (k_i - r_i * a_i(t)) * 
                        \\mathcal{S}_i(c_3 a_e(t) - c_4 a_i(t) + I_{ext_j}(t))

        &\\mathcal{S}(x) = \\frac{1}{1 + exp(- a(x - \\theta))} - \\frac{1}{1 + exp(a\\theta)} 

    ST refers to neuron state, members of ST are listed below:
    
    =============== ================= =========================================================
    **Member name** **Initial Value** **Explanation**
    --------------- ----------------- ---------------------------------------------------------
    a_e             0.1               The proportion of excitatory cells firing per unit time.
    
    a_i             0.05              The proportion of inhibitory cells firing per unit time.
    
    input_e         0.                External input to excitatory cells.
                             
    input_i         0.                External input to inhibitory cells.
    =============== ================= =========================================================
    
    Note that all ST members are saved as floating point type in BrainPy, 
    though some of them represent other data types (such as boolean).

    Args:
        c1 (float): Weight from E-neurons to E-neurons.
        c2 (float): Weight from I-neurons to E-neurons.
        c3 (float): Weight from E-neurons to I-neurons.
        c4 (float): Weight from I-neurons to I-neurons.
        k_e (float): Model parameter, control E-neurons' refractory period together with r_e.
        k_i (float): Model parameter, control I-neurons' refractory period together with r_i.
        tau_e (float): Time constant of E-neurons' activity.
        tau_i (float): Time constant of I-neurons' activity.
        r_e (float): Model parameter, control E-neurons' refractory period together with k_e.
        r_i (float): Model parameter, control I-neurons' refractory period together with k_i.
        slope_e (float): E-neurons' sigmoid function slope parameter.
        slope_i (float): I-neurons' sigmoid function slope parameter.
        theta_e (float): E-neurons' sigmoid function phase parameter.
        theta_i (float): I-neurons' sigmoid function phase parameter.
        mode (str): Data structure of ST members.

    References:
        .. [1] Wilson, Hugh R., and Jack D. Cowan. "Excitatory and inhibitory 
               interactions in localized populations of model neurons." 
               Biophysical journal 12.1 (1972): 1-24.

    
    """

    ST = bp.types.NeuState(
        {'a_e': 0.1, 'a_i': 0.05, 'input_e': 0., 'input_i': 0.}
    )

    def mysigmoid(x, a, theta):
        return 1 / (1 + np.exp(- a * (x - theta))) - 1 / (1 + np.exp(a * theta))

    @bp.integrate
    def int_a_e(a_e, _t, a_i, I_ext_e):
        return (- a_e + (k_e - r_e * a_e) *
                mysigmoid(c1 * a_e - c2* a_i + I_ext_e, slope_e, theta_e))/tau_e

    @bp.integrate
    def int_a_i(a_i, _t, a_e, I_ext_i):
        return (- a_i + (k_i - r_i * a_i) *
                mysigmoid(c3 * a_e - c4 * a_i + I_ext_i, slope_i, theta_i))/tau_i

    def update(ST, _t):
        a_e = int_a_e(ST['a_e'], _t, ST['a_i'], ST['input_e'])
        a_i = int_a_i(ST['a_i'], _t, ST['a_e'], ST['input_i'])
        ST['a_e'] = a_e
        ST['a_i'] = a_i

    def reset(ST):
        ST['input_e'] = 0
        ST['input_i'] = 0

    
    if mode == 'scalar':
        return bp.NeuType(name='WilsonCowan_neuron',
                          ST=ST,
                          steps=(update, reset),
                          mode=mode)
    elif mode == 'vector':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    elif mode == 'matrix':
        raise ValueError("mode of function '%s' can not be '%s'." % (sys._getframe().f_code.co_name, mode))
    else:
        raise ValueError("BrainPy does not support mode '%s'." % (mode))
