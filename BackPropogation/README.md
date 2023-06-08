# Tutorial : Backpropagation

Backpropagation is used to adjust the weights and biases of the neural network. It is primarily employed to minimize the difference between the predicted output and the expected output. The Backpropagation algorithm is divided into two phases.

- __Forward Propagation__: During this step, the input data is fed into the neural network, and activations are calculated for each layer. This process is repeated until reaching the output layer.
- __Backward Propagation__: Once the forward propagation is complete and the network has produced an output, the algorithm calculates the difference between the predicted output and the expected output. This difference is known as the "error" or "loss." The backpropagation algorithm then propagates this error backward through the network, starting from the output layer and moving towards the input layer. The error is distributed back to the neurons in each layer, and the algorithm calculates how much each neuron's activation contributed to the overall error. This information is used to adjust the weights and biases of the connections in the network, aiming to minimize the error. Backpropagation explained in below image

![030819_0937_BackPropaga1](https://github.com/paritosh1505/ERAV1/assets/6744935/88535e8f-f0ad-4ed2-9abc-f1507fe9eb92)


# How Backpropagation works
In this example, we will explain how the backpropagation algorithm works. For this scenario, we will use a neural network architecture, which is displayed on the following image.

![BackProp](https://github.com/paritosh1505/ERAV1/assets/6744935/6e93bcd6-17da-4040-a86d-01c297dc6312)

 ## Step 1: nitialize initial inputs
In the first step, we are initializing the initial inputs, which are explained below:
- __Target Variables__: Here, we are using two target variables, t1 and t2, which are initialized as 0.5 (t1) and 0.5 (t2).
- __Input Variables__: For this neural architecture, we are using two input variables, i1 and i2, which are initialized as 0.05 (i1) and 0.1 (i2).
- __Weight Variables__: As shown in the above image, there are eight weight variables, w1, w2, w3, w4,w5,w6w7 and w8, which are initialized as 0.15 (w1), 0.2 (w2), 0.23 (w3), 0.3 (w4),0.4(w5)	,0.45(w6),	0.5(w7) and	0.55(w8)
![image](https://github.com/paritosh1505/ERAV1/assets/6744935/7e2557d5-89ec-4a43-a386-7b2dc09c03a7)
.


## Step 2 Forward propogation
As explained above, in forward propagation, we feed input data into the neural network. Similarly, here we also feed input data into the neural network and find the value of h1 (the first neuron present in the hidden layer) using the same approach. Additionally, we calculate h2 (the second neuron present in the hidden layer).

```bash
  hidden_layer = weight_variable*input_variable
```
if we follow above example h1 = (I2*W2) + (I1*w1) and h2 = (I1*w3)+(I2*w4) hence h1 and h2 will become .0275 and .04250. As shown in the above image, h1 and h2 are associated with the activation function. Here, we are using the softmax activation function. The formula for softmax is as follows:
```bash
  softmax(x) = 1/(1+Exp(x))
```
softmax of h1 is defined as a_h1 which is equal to 0.506 and a_h2 is 0.5106

Now, our next step is to calculate o1 and o2 using the same approach as we did for calculating h1 and h2. Additionally, o1 is associated with the activation function called a_o1, and o2 is associated with the activation function called a_o2.
After calcluating a_o1 and a_o2 next task is calcuatin error value which is E1 and E2. Here error is defined as

```bash
  Error = 0.5*(target-a_0)**2
```
hence value of E1 and E2 will be 0.005664	0.00851
and total error value will be E1+E2 = 0.0141
Complete forward propogation can be summed up in following excel file

![Forward](https://github.com/paritosh1505/ERAV1/assets/6744935/799a86c1-d474-4cd3-8a7a-1eba33fbd352)

