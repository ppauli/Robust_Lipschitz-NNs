# -*- coding: utf-8 -*-
"""
@author: ppauli
"""

import torch
import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()

def build_T(weights,biases,net_dims):
    x = {
        'n_x': matlab.int64([net_dims[0]]),
        'n_h': matlab.int64([net_dims[1]]),
        'n_y': matlab.int64([net_dims[2]]),        
    }
    
    parameters = {}
    for i in range(len(weights)):
        parameters.update({
            'W{:d}'.format(i): matlab.double(np.array(weights, dtype=np.object)[i].tolist()),
            })
        parameters.update({
            'b{:d}'.format(i): matlab.double(np.array(biases, dtype=np.object)[i].tolist()),
            })    
    Lip = eng.calculate_Lipschitz(parameters,x)
    return Lip
        
def solve_SDP(parameters,T,net_dims,rho,mu,ind_Lip,L_des):
    x = {
        'T': T,
        'n_x': matlab.int64([net_dims[0]]),
        'n_h': matlab.int64([net_dims[1]]),
        'n_y': matlab.int64([net_dims[2]]),
        'rho': matlab.double([rho]),
        'mu': matlab.double([mu]),
        'ind_Lip': matlab.int64([ind_Lip]),
        'L_des': matlab.double([L_des])
    }
        
    parameters = eng.solve_sdp(parameters,x)
    return parameters

def initialize_parameters(weights,biases):
    parameters = {}
    for i in range(len(weights)):
        parameters.update({
            'W{:d}'.format(i): matlab.double(np.array(weights, dtype=np.object)[i].tolist()),
            })
        parameters.update({
            'W{:d}_bar'.format(i): matlab.double(np.array(weights, dtype=np.object)[i].tolist()),
            })
        parameters.update({
            'b{:d}'.format(i): matlab.double(np.array(biases, dtype=np.object)[i].tolist()),
            })
        parameters.update({
            'Y{:d}'.format(i): matlab.double(torch.zeros(weights[i].shape).tolist()),
            })
        parameters.update({
            'Yb{:d}'.format(i): matlab.double(torch.zeros(biases[i].shape).tolist()),
            })
    return parameters