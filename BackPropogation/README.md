# Tutorial : How Backpropagation work

Backpropagation is used to adjust the weights and biases of the neural network. It is primarily employed to minimize the difference between the predicted output and the expected output. The Backpropagation algorithm is divided into two phases.

- __Forward Propagation__: During this step, the input data is fed into the neural network, and activations are calculated for each layer. This process is repeated until reaching the output layer.
- __Backward Propagation__: Once the forward propagation is complete and the network has produced an output, the algorithm calculates the difference between the predicted output and the expected output. This difference is known as the "error" or "loss." The backpropagation algorithm then propagates this error backward through the network, starting from the output layer and moving towards the input layer. The error is distributed back to the neurons in each layer, and the algorithm calculates how much each neuron's activation contributed to the overall error. This information is used to adjust the weights and biases of the connections in the network, aiming to minimize the error.




![BackProp](https://github.com/paritosh1505/ERAV1/assets/6744935/6e93bcd6-17da-4040-a86d-01c297dc6312)
