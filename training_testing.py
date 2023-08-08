# TRAINING

# define a function for training
MSE = nn.MSELoss()

nEpochs = 10 ####
def train(model, dataloder, loss_func, optimizer, nEpochs):
    for epoch in range(nEpochs):
        # print("Epoch " + str(epoch) + '/' + str(nEpochs))
        for x_in_, y_out_ in dataloder:
            x_res_init = torch.zeros(x_in_.shape[0], dim_res) # set initial states
            x_out_ = model(x_res_init, x_in_)
            loss = loss_func(x_out_,y_out_)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

# TESTING
# return predicted result, and ground truth
def test(model, dataloder):
    x_out_list = []
    y_out_list = []
    model.eval()
    with torch.no_grad():
        for x_in_, y_out_ in dataloder:
            x_res_init = torch.zeros(x_in_.shape[0], dim_res) # set initial states
            x_out_ = model(x_res_init, x_in_)
            x_out_list.append(x_out_)
            y_out_list.append(y_out_)
    x_out_list = torch.cat(x_out_list, dim=0)
    x_out_np = x_out_list.detach().numpy()
    y_out_list = torch.cat(y_out_list, dim=0)
    y_out_np = y_out_list.detach().numpy()
    return x_out_np, y_out_np
