# Tutorial : Backpropagation

Backpropagation is used to adjust the weights and biases of the neural network. It is primarily employed to minimize the difference between the predicted output and the expected output. The Backpropagation algorithm is divided into two phases.

- __Forward Propagation__: During this step, the input data is fed into the neural network, and activations are calculated for each layer. This process is repeated until reaching the output layer.
- __Backward Propagation__: Once the forward propagation is complete and the network has produced an output, the algorithm calculates the difference between the predicted output and the expected output. This difference is known as the "error" or "loss." The backpropagation algorithm then propagates this error backward through the network, starting from the output layer and moving towards the input layer. The error is distributed back to the neurons in each layer, and the algorithm calculates how much each neuron's activation contributed to the overall error. This information is used to adjust the weights and biases of the connections in the network, aiming to minimize the error. Backpropagation explained in below image

![030819_0937_BackPropaga1](https://github.com/paritosh1505/ERAV1/assets/6744935/88535e8f-f0ad-4ed2-9abc-f1507fe9eb92)


# How Backpropagation works
In this example, we will explain how the backpropagation algorithm works. For this scenario, we will use a neural network architecture, which is displayed on the following image.

![BackProp](https://github.com/paritosh1505/ERAV1/assets/6744935/6e93bcd6-17da-4040-a86d-01c297dc6312)

 ## Step 1: Initialize initial inputs
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

## Step 3 Backwrd Propogation
Now, in backpropagation, the error which we calculate during the forward propagation step will propagate backward, starting from the output layer and moving towards the input layer. This is basically used for minimizing the error. Let's first see the backpropagation formula for each layer

We continue where we stops in forward propogation i.e. calcuating total error. Our firs step is calcaute gradients of the individual errors with respect to the weights in order to update the weights and minimize the total error. By applying the chain rule of calculus. hence partial derivate of total error value with weight w5 is 

```bash
  ∂E_total/∂w5 = ∂(E1 + E2)/∂w5
  ∂E_total/∂w5 = ∂E1/∂w5 (As w5 and E2 is not dependent on each other)
  ∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5 (Chaining Rule)
  Let 
  D1 = ∂E1/∂a_o1
  D2 = ∂a_o1/∂o1
  D3 = ∂o1/∂w5
  Hence
  D1 =  ∂(½ * (t1 - a_o1)²)/∂a_o1 = (a_01 - t1) 
  D2 =  ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)
  D3 =  a_h1
  After combining D1,D2,D3 we will get
  ∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1
```
As we can see in the above example, we calculate the partial derivatives of the error with respect to the weights  w6, w7 and w8.

```bash
 ∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2
 ∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1
 ∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2
```
Our next step should be to calculate the partial derivatives of the error with respect to the activation functions a_h1 and a_h2. Let's first calculate the partial derivative of the error with respect to a_h1.

```bash
 ∂E_total/∂a_h1 = ∂E1/∂a_h1= ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂a_h1 (Chaining Rule)
 ∂E1/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5 - eq 1
```
Since a_h1 is also dependent on Error 2 hecne final partial derivative would be

```bash
 ∂E_total/∂a_h1 = ∂E2/∂a_h1 = ∂E2/∂a_o2*∂a_o2/∂o2*∂o2/∂a_h1 (Chaining Rule)
 ∂E2/∂a_h1 = (a_02 - t2) * a_o2 * (1 - a_o2) * w7 - eq 2
```
Hence total error value w.r.t a_h1 will be 

```bash
 by adding eq1 and eq2
 ∂E_total/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7--eq 3
```
Similarly partial derivate value w.r.t a_h2 is 

```bash
 ∂E_total/∂a_h2 = (a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8!
```
in final step we have to calculate parial derivate wr.r.t w1,w2,w3 and w4

```bash

 ∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
 Let D1 =  ∂E_total/∂a_h1
 D2 = ∂a_h1/∂h1 
 D3 =  ∂h1/∂w1
 D1 can be fetch from eq 3
 D2 = a_h1 * (1 - a_h1)
 D3 = I1 as h1 = (I2*W2) + (I1*w1)
 Hence 
 ∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
```
Similary partial derivative w.r.t w2,w3 and w4
```bash
∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2
∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1
∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2
```
By using above formula we can calualte partial derivative of error w.r.t w1,w2,w3 and w4 whih is shown in below image. (Below image is not complee screenshot it is the coninuation of abve excel file screenshot)

![Backward](https://github.com/paritosh1505/ERAV1/assets/6744935/ed3ca125-1a71-4ec1-8424-e1b16b9b6eb0)

## Step 4 Weight Update and forward Propogation
Our next step is updating the weight . Formula for updating the weight is

```bash
w_new = w_old - eta * (∂E_total/∂w)-- eq4
```
- w_old : The term "w_old" represents the old (previous) value of the weight w. It is the weight value before applying the weight update based on the gradient descent algorithm.
- eta : It is called learning rate.The learning rate is a crucial parameter as it affects the convergence and training speed of the neural network. A larger learning rate can lead to faster convergence, but it may also risk overshooting the optimal solution. On the other hand, a smaller learning rate can result in slower convergence but potentially better precision. *__For this scenario we are using learning rate = 1__* We will change the learning rate to different value in later section

Hence, first, we update the weight using equation 4. Then, we perform forward propagation again, similar to what we did in the section "*Step 2: Forward Propagation*." However, in this step, we do not use the old weight; instead, we utilize the updated weight calculated using equation 4.

Hence updated weight value along with old weight is shown below
![Forward](https://github.com/paritosh1505/ERAV1/assets/6744935/17deab69-b69e-4a96-998f-c8cea06f7d46)

After calculating the total error during forward propagation, we use backpropagation to compute the gradients, which follows a similar approach as the one described above. The resulting gradients are then used to update the weights and biases of the network, typically using an optimization algorithm like gradient descent.

![image](https://github.com/paritosh1505/ERAV1/assets/6744935/1231303f-f620-40c7-99f6-a9b039e8da29)

Again, we need to update the weights, perform forward propagation, calculate gradients, and then update the weights again. This step can be repeated until the stopping criteria or convergence scenario are met.folloing is the updated value for 20 epochs

*__Forward propagation calculation till 20 epochs.__*

![image](https://github.com/paritosh1505/ERAV1/assets/6744935/dc102d56-f10b-462c-84f7-04a9cf79267e)

*__Backward propagation calculation till 20 epochs.__*

![image](https://github.com/paritosh1505/ERAV1/assets/6744935/57dbdede-8d2a-4957-9f6e-fb566de42eb4)

*__Loss Graph for Above scenario till 100 epochs__*

![image](https://github.com/paritosh1505/ERAV1/assets/6744935/575fc188-7ea8-4e6f-9aef-6d8c4b3f9398)

### Loss Graph for Different Learning rate__

-*__Scenario 1: When learning rate is 0.1__*

As we can see, for a learning rate of 0.1, the loss value is decreasing very slowly, which is not an optimized scenario.

![image](https://github.com/paritosh1505/ERAV1/assets/6744935/0970ab2a-72ac-40a2-8ea4-053c69f518ac)

-*__Scenario 2: When learning rate is 0.2__*

As we can see, for a learning rate of 0.2, the loss value is decreasing very slowly, which is not an optimized scenario.

![image](https://github.com/paritosh1505/ERAV1/assets/6744935/5b0de5d8-f9cc-41c5-ad4d-c87125bbc35c)

-*__Scenario 3: When learning rate is 0.5__*

Although the loss value is decreasing much better for a learning rate of 0.5 compared to when the learning rate is 0.1 and 0.2, there is still a chance for further improvement, which we will cover in the next scenario.

![image](https://github.com/paritosh1505/ERAV1/assets/6744935/98ec0624-5b54-4be3-9f28-45f530fc3de1)

-*__Scenario 3: When learning rate is 0.8__*

 Now loss is decreasing at much faster rate 
 
 ![image](https://github.com/paritosh1505/ERAV1/assets/6744935/a383e741-4a5f-436b-b1e2-897f4c3561ad)
 
 -*__Scenario 4: When learning rate is 1__*
 
 ![image](https://github.com/paritosh1505/ERAV1/assets/6744935/b2ec67bb-c575-4a7d-bf0f-140598bd43f6)

 -*__Scenario 5: When learning rate is 2__*
 
As we can see increasing when we are increasing the learning rate loss value is coming to close to 0 at much faster rate

![image](https://github.com/paritosh1505/ERAV1/assets/6744935/33b8045b-138d-4698-92fa-bb11b71a5fc4)

### Conclusion
Backpropagation is an algorithm used to adjust the weights and biases of a neural network in order to minimize the difference between predicted and expected outputs.
During forward propagation, input data is fed into the neural network, and activations are calculated for each layer until reaching the output layer. The predicted output is obtained, and the error or loss between the predicted and expected output is determined.
In backward propagation, the error is propagated backward through the network, starting from the output layer and moving towards the input layer. The algorithm calculates how much each neuron's activation contributed to the overall error and uses this information to adjust the weights and biases of the connections in the network.The backpropagation process involves calculating partial derivatives of the error with respect to the weights and activation functions at each layer. These derivatives are used to update the weights and biases through an optimization algorithm like gradient descent.
By iteratively performing forward propagation, calculating gradients, and updating the weights, the neural network learns to minimize the error and improve its predictions. This iterative process continues until a stopping criteria or convergence condition is met.
