lin_act = lambda x:x

dim_res_0 = 2
X_in = torch.randn((N_total, L, dim_in))
ESN_0 = esn(dim_in=dim_in, dim_res=dim_res_0, dim_out=dim_out, activation=lin_act, len_seq=L)

# gamma = 0.95
# w_res_0 = np.random.randn(dim_res_0, dim_res_0)
# w_res_0 = w_res_0 / np.linalg.norm(w_res_0, ord = 2) * gamma

# using a fixed w_res_0

c = 1   # c < \sqrt(2) to make stable, a scaling factor
w_res_0 = np.array([[.5, .5], [-.5, .5]]) * c
w_in_0 = np.array([[.5], [.5]])
b_in_0 = np.array([0, 0])
w_out_0 = np.array([[.5, .5]])
b_out_0 = np.array([0])

np2t = lambda x: torch.tensor(x).type(torch.FloatTensor)

ESN_0.set_w_res(np2t(w_res_0))
ESN_0.set_wb_in(np2t(w_in_0), np2t(b_in_0))
ESN_0.set_wb_out(np2t(w_out_0), np2t(b_out_0))

# by linear

x_res_init = torch.zeros(X_in.shape[0], dim_res_0) # set initial states

with torch.no_grad():
    Y_out = ESN_0(x_res_init,X_in).detach()

from torch.utils.data import TensorDataset, DataLoader

# dataloder for training
ds = TensorDataset(X_in[:N_train], Y_out[:N_train])
ds_test = TensorDataset(X_in[N_train:], Y_out[N_train:])
batchsize = 64
dl = DataLoader(ds, batch_size=batchsize, shuffle = True)
dl_test = DataLoader(ds_test, batch_size=batchsize) # do not shuffle.

# Y_test
X_test = X_in[N_train:]
Y_test = Y_out[N_train:]


### if you want to visualize input and output, include the following codes

## input X_in
# import matplotlib.pyplot as plt
# for i in range(4):
#    plt.figure()
#    plt.plot(X_in[i],'b')

## output Y_out
#for i in range(4):
#    plt.figure()
#    plt.plot(Y_out[i],'b')
