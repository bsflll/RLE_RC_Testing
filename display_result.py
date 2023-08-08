import matplotlib.pyplot as plt

dim_res_list = list(range(1, 11))

act_list = ['linear', 'tanh', 'relu']
# plot the graph for each model
for i, model_errors in enumerate(err_list):
    plt.plot(dim_res_list, model_errors, label=f'Model: ' + act_list[i]) #  {i+1}

# add labels and title
plt.xlabel('Reservoir Dimensions (dim_res)')
plt.ylabel('Error (L2 Loss)')
plt.title('Error of Reservoir Dimensions for Three Models with Different Activation Functions')
plt.legend()

# display
plt.show()

np.save("hereismee.npy", err_list)
