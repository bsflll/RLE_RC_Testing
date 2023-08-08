# convert Y_test to numpy and
# Y_test_np = Y_test.detach().numpy()
#compute error
def L2(y , y_target):
    return np.mean(np.sum((y - y_target)**2,axis=1)) #matrix -> compute square



# CALC ERROR
err_list = []
act_list = [lin_act, F.tanh, F.relu]
from  tqdm import tqdm

for act in act_list:
    err_list_act = []
    for dim_res in range(1, 11):
        err = 0
        err_list_list = []
        for i in tqdm(range(1,11)):
            #dim_res = 4
            ESN_test = esn(dim_in=dim_in, dim_res=dim_res, dim_out=dim_out, activation=act, len_seq=L) # how
            #ESN_test.set_w_res(torch.randn(dim_res, dim_res) / np.sqrt(dim_res)* 0.1) 
            # we might want to try different initialization
            ESN_test.freeze()
            opt = optim.Adam(ESN_test.parameters())
            train(ESN_test, dl, MSE, opt, nEpochs)
            x_out_np, y_out_np = test(ESN_test, dl_test)
            err = L2(x_out_np, y_out_np)
            err_list_list.append(err)
        err_list_act.append(err/10) # not a number will crash
    err_list.append(err_list_act)
