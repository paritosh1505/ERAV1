
### CODE 1: BASIC SKELETON
CODE

**Target:**

Create the basic neural network architecture for MNIST dataset

**Results:**

- Parameters: 119,392
- Best Train Accuracy: 99.32
- Best Test Accuracy: 98.51

**Analysis:**

Extremely large paramter for simple dataset(MNIST)
Model is over-fitting, we need to change our model and make it lighter

### Code 2 : Define Lighter architecture

**Target:**

Create lighter model architecture

**Results:**

- Parameters :4990
- Best Train Accuracy: 96.84
- Best Test Accuracy: 97.16

**Analysis:**

Model has 8790 paramter since train accuracy is less than test accuracy hence there is a chances of improvement


### Code 3 : ADD batch normalization



**Target:**

To improve the performance of model we are adding BN.

**Results:**

- Parameters :5150
- Best Train Accuracy: 94.06
- Best Test Accuracy: 98.07

**Analysis:**

Model has 5150 paramter since train accuracy is less than test accuracy hence there is a chances of improvement but ccuracy is decreasing here so we need to add some more optimization technique


### Code 4 : ADD capacity of network

**Target:**

Add more layer to current architecture

** Results:**

- Parameters :7904
- Best Train Accuracy: 98.60
- Best Test Accuracy: 99.39

**Analysis:**

Model has 7904 paramter after adding capacity of model test accuracy is improved from 98.07 to 99.39 as we know when We can also increase the capacity of the model by adding a layer after GAP! here we did the same we added one extra layer after GAP and also increases the paramter

### Code 5 : Final Netowrk

**Target:**

Adding image augumentation in the architecture jsut to increase the variation in the image so that training become difficult and hencce test accuracy will improve

**Results:**

- Parameters :7904
- Best Train Accuracy: 94.06
- Best Test Accuracy: 99.41

**Analysis:**

Final Model has 7904 paramter since train accuracy is less than test accuracy hence there is a chances of improvement and here weacheived the accuracy of model greater than 99.4 with paramter less than 8000 and less than 15 epochs

**Model Summary**
```bash
  ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
              ReLU-2           [-1, 32, 26, 26]               0
       BatchNorm2d-3           [-1, 32, 26, 26]              64
         Dropout2d-4           [-1, 32, 26, 26]               0
            Conv2d-5            [-1, 8, 26, 26]             256
              ReLU-6            [-1, 8, 26, 26]               0
       BatchNorm2d-7            [-1, 8, 26, 26]              16
         Dropout2d-8            [-1, 8, 26, 26]               0
            Conv2d-9           [-1, 10, 24, 24]             720
             ReLU-10           [-1, 10, 24, 24]               0
      BatchNorm2d-11           [-1, 10, 24, 24]              20
        Dropout2d-12           [-1, 10, 24, 24]               0
        MaxPool2d-13           [-1, 10, 12, 12]               0
           Conv2d-14           [-1, 16, 12, 12]           1,440
             ReLU-15           [-1, 16, 12, 12]               0
      BatchNorm2d-16           [-1, 16, 12, 12]              32
        Dropout2d-17           [-1, 16, 12, 12]               0
           Conv2d-18           [-1, 10, 12, 12]             160
             ReLU-19           [-1, 10, 12, 12]               0
      BatchNorm2d-20           [-1, 10, 12, 12]              20
        Dropout2d-21           [-1, 10, 12, 12]               0
           Conv2d-22           [-1, 10, 10, 10]             900
             ReLU-23           [-1, 10, 10, 10]               0
      BatchNorm2d-24           [-1, 10, 10, 10]              20
        Dropout2d-25           [-1, 10, 10, 10]               0
           Conv2d-26             [-1, 16, 8, 8]           1,440
             ReLU-27             [-1, 16, 8, 8]               0
      BatchNorm2d-28             [-1, 16, 8, 8]              32
        Dropout2d-29             [-1, 16, 8, 8]               0
           Conv2d-30             [-1, 16, 6, 6]           2,304
             ReLU-31             [-1, 16, 6, 6]               0
      BatchNorm2d-32             [-1, 16, 6, 6]              32
        Dropout2d-33             [-1, 16, 6, 6]               0
        AvgPool2d-34             [-1, 16, 1, 1]               0
           Conv2d-35             [-1, 10, 1, 1]             160
================================================================
Total params: 7,904
Trainable params: 7,904
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.21
Params size (MB): 0.03
Estimated Total Size (MB): 1.24
----------------------------------------------------------------
```



