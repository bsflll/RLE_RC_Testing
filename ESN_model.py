from torch import optim

import torch
import torch.nn as nn
import torch.nn.functional as F

class esn(nn.Module):
    def __init__(self, dim_in, dim_res, dim_out, len_seq, activation = F.tanh, x_res_init = 0):
        #dim_in : The number of expected features in the input x
        #dim_res: The number of features in the x_res
        #dim_out: output dim
        #len_seq: L
        #x_res_init: of shape (dim_res,)
        super().__init__()
        self.dim_in = dim_in
        self.dim_res = dim_res
        self.dim_out = dim_out
        self.len_seq = len_seq # length of sequence
        self.activation = activation
        self.lin_in = nn.Linear(dim_in, dim_res) # W_in: linear layer for reading out the outputs
        self.lin_res = nn.Linear(dim_res, dim_res, bias=False) # W_res: linear layer for reading out the outputs
        self.lin_out = nn.Linear(dim_res, dim_out) # W_out: linear layer for reading out the outputs by default it has bias, symmetrical as lin_in
        self.x_res = x_res_init


### TODO:
#### set the weights,
#### freeze, unfreeze the weights
#### set_x_res
## "memory"..
#######
# ----- def read_w_res def read_w_in
    def read_w_res(self):
        return self.lin_res.weight.detach()

    def read_wb_in(self):
        return self.lin_in.get_parameter('weight'), self.lin_in.get_parameter('bias')

    def set_w_res(self, w_res):
        with torch.no_grad():
            self.lin_res.weight = nn.Parameter(w_res)

    def set_wb_in(self, w_in, b_in):
        with torch.no_grad():
            self.lin_in.weight = nn.Parameter(w_in)
            self.lin_in.bias = nn.Parameter(b_in)

    def set_wb_out(self, w_out, b_out):
        with torch.no_grad():
            self.lin_out.weight = nn.Parameter(w_out)
            self.lin_out.bias = nn.Parameter(b_out)

    def freeze(self):
        for name, para in self.lin_in.named_parameters():
            para.requires_grad = False
        self.lin_res.requires_grad_(False) # by default, linear model is trainable

    def unfreeze(self):
        for name, para in self.lin_in.named_parameters():
            para.requires_grad = True
        self.lin_res.requires_grad_(True)

    def forward(self, x_res_init, x_in):
        # x_res_init: batchsize x dim_in  : initial state
        # x_in: batchsize x len_seq x dim_in
        # x_res: batchsize x len_seq x dim_res
        # x_out: batchsize x len_seq x dim_out
        batchsize = x_in.shape[0]
        len_seq = x_in.shape[1]
        x_res_list = []# = torch.zeros(batchsize, len_seq, self.dim_res)
        #x_out = torch.zeros(batchsize, len_seq, self.dim_out)
        # compute x_res
        for m in range(self.len_seq):
            if m == 0:
                x_res = x_res_init
            else:
                x_res = self.activation(self.lin_res(x_res_list[-1]) + self.lin_in(x_in[:, m-1, :]))
            x_res_list.append(x_res)
        # compute x_out
        x_out = [self.lin_out(x_res) for x_res in x_res_list]
        x_out = torch.stack(x_out, dim=1)
        return x_out
