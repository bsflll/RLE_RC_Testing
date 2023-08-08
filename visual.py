# visualize data applying signal processing 
# just to show the result


impulse_input = torch.zeros(1, L, 1)
impulse_input[0, 0, 0] = 1
x_res_init = torch.zeros(impulse_input.shape[0], dim_res_0) # set initial states

with torch.no_grad():
    impulse_response = ESN_0(x_res_init, impulse_input).detach()



impulse_response = impulse_response.reshape(-1)
freq_response = np.fft.fft(impulse_response)




import matplotlib.pyplot as plt
plt.figure()
plt.plot(impulse_response)
plt.figure()
plt.plot(np.abs(freq_response))
