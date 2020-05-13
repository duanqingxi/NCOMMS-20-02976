import torch
import torch.nn as nn

class LIFNeuron(nn.Module):
    '''
    LIF Neuron model, parameters are extracted from experimental data.
    Rd: device HRS
    Cm: parallel capacitor
    Rs: series resistance
    Vth: threshold voltage of Neuron device
    V_reset: reset voltage of Neuron
    v: membrane potential
    dt: simulation time step 
    '''
    def __init__(self, batch_size, dim_in, Rd, Cm, Rs, Vth, V_reset, dt):
        super().__init__()
        self.batch_size = batch_size
        self.dim_in = dim_in
        self.rd = Rd
        self.cm = Cm
        self.rs = Rs
        self.vth = Vth
        self.v_reset = V_reset
        self.v = torch.full([self.batch_size, self.dim_in], self.v_reset)
        self.dt = dt

        self.tau_in = 1/(self.cm*self.rs)
        self.tau_lk = 1/(self.cm)*(1/self.rd + 1/self.rs) 
    
    @staticmethod
    def soft_spike(x):
        a = 2.0
        return torch.sigmoid_(a*x)
    
    def spiking(self):
        if self.training == True:
            spike_hard = torch.gt(self.v, self.vth).float()
            spike_soft = self.soft_spike(self.v - self.vth)
            v_hard = self.v_reset*spike_hard + self.v*(1 - spike_hard)
            v_soft = self.v_reset*spike_soft + self.v*(1 - spike_soft)
            self.v = v_soft + (v_hard - v_soft).detach_()
            return spike_soft + (spike_hard - spike_soft).detach_()
        else:
            spike_hard = torch.gt(self.v, self.vth).float()
            self.v = self.v_reset*spike_hard + self.v*(1 - spike_hard)
            return spike_hard


    def forward(self, v_inject):
        '''
        Upgrade membrane potention every time step by differantial equation.
        '''
        self.v += (self.tau_in*v_inject - self.tau_lk*self.v) * self.dt
        return self.spiking()
    
    def reset(self):
        '''
        Reset the membrane potential to reset voltage.
        '''
        self.v = torch.full([self.batch_size, self.dim_in], self.v_reset)

