import torch
import torch.nn as nn
import math


class MAC_Crossbar(nn.Module):
    '''
    A linear layer which performs multuiply and accumulation as a crossbar.
    Input: a batch of m-dimentional input vector.
    output: a batch of n-dimentional weighted sum of input.

    the weights in this layer will be quantized to q_bit if .Gquant_() method is called during backward process
    '''
    def __init__(self, dim_in, dim_out, G_min, G_max, q_bit):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.G_min = G_min
        self.G_max = G_max
        self.lsb = 1.0e-6
        self.weight = nn.Parameter(torch.zeros(dim_in, dim_out))
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)
        self.q_bit = q_bit
        self.w_max = 1.0
        self.w_min = 0.0

    def forward(self, input_vector):
        output = input_vector.mm(self.weight)
        return output
    
    def Gquant_(self):
        self.w_max = min(self.weight.max().item(), self.G_max)
        self.w_min = max(self.weight.min().item(), self.G_min)
        self.lsb = (self.w_max-self.w_min)/math.pow(2,self.q_bit)
        self.weight.data = torch.clamp_(self.weight, self.G_min, self.G_max)
        self.weight.data = ((self.weight-self.w_min)/self.lsb).round()*self.lsb + self.w_min
    
class TIA_Norm(nn.Module):
    '''
    Normolize the column output current to moderate voltage.
    '''
    def __init__(self, max_v, min_v, amp):
        super().__init__()
        self.max_v = max_v
        self.min_v = min_v
        self.amp = amp
        self.span = self.max_v - self.min_v

    def forward(self, input_vector):
        output_vector = self.amp*(input_vector - self.min_v)/self.span
        output_vector = torch.clamp_(output_vector, -2.0, 2.0)
        return output_vector